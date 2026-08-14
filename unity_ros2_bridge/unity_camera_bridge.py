import json
import math
import socket
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped, Twist, TwistWithCovarianceStamped
from mfdf_ros2_msgs.msg import SimulatedSource, SimulatedSourceArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from spot_msgs.msg import Feedback, PowerState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

HEADER_SIZE = 4
CAMERA_MAGIC = b"UCAM"
CAMERA_VERSION = 1
CAMERA_HEADER = struct.Struct("<4sHHIHH" + "f" * 13 + "II")
LIDAR_MAGIC = b"ULDR"
LIDAR_VERSION = 1
LIDAR_HEADER = struct.Struct("<4sHHIHHfffffffffffff")
POINT_XYZ = struct.Struct("<fff")


class InvalidFrameError(Exception):
    pass


@dataclass(frozen=True)
class RgbdFrame:
    sequence: int
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near_clip: float
    far_clip: float
    mount: Tuple[float, float, float, float, float, float, float]
    jpeg: bytes
    depth: bytes


def receive_exact(client: socket.socket, size: int) -> Optional[bytes]:
    data = bytearray()
    while len(data) < size:
        try:
            chunk = client.recv(size - len(data))
        except socket.timeout:
            continue
        if not chunk:
            if not data:
                return None
            raise InvalidFrameError(
                f"connection closed after {len(data)} of {size} bytes"
            )
        data.extend(chunk)
    return bytes(data)


class UnitySpotBridge(Node):
    """Expose the Unity Spot simulation through spot_driver-compatible ROS APIs."""

    def __init__(self) -> None:
        super().__init__("unity_spot_bridge")

        self.declare_parameter("listen_host", "0.0.0.0")
        self.declare_parameter("camera_port", 50051)
        self.declare_parameter("control_port", 50052)
        self.declare_parameter("lidar_port", 50053)
        self.declare_parameter("spot_name", "spot")
        self.declare_parameter("camera_name", "frontleft")
        self.declare_parameter("camera_topic", "camera/image/compressed")
        self.declare_parameter(
            "camera_frame_id", "spot/camera/frontleft_optical_frame"
        )
        self.declare_parameter("vision_frame_id", "spot/vision")
        self.declare_parameter("odom_frame_id", "spot/odom")
        self.declare_parameter("body_frame_id", "spot/body")
        self.declare_parameter("lidar_frame_id", "spot/lidar")
        self.declare_parameter("tf_root", "odom")
        self.declare_parameter("map_frame_id", "map")
        self.declare_parameter("lamp_base_frame_id", "lamp_base_link")
        self.declare_parameter(
            "radiation_source_topic", "/simulation/radiation_sources"
        )
        self.declare_parameter("lamp_mount_xyz", [0.0, 0.0, 0.25])
        self.declare_parameter("lamp_mount_rpy", [math.pi, 0.0, 0.0])
        self.declare_parameter("max_payload_bytes", 5 * 1024 * 1024)
        self.declare_parameter("max_lidar_payload_bytes", 1024 * 1024)
        self.declare_parameter("command_timeout_seconds", 2.0)
        self.declare_parameter("rgbd_publish_rate_hz", 1.0)

        self._listen_host = str(self.get_parameter("listen_host").value)
        self._camera_port = int(self.get_parameter("camera_port").value)
        self._control_port = int(self.get_parameter("control_port").value)
        self._lidar_port = int(self.get_parameter("lidar_port").value)
        self._spot_name = str(self.get_parameter("spot_name").value).strip("/")
        self._camera_name = str(self.get_parameter("camera_name").value).strip("/")
        self._camera_frame_id = str(
            self.get_parameter("camera_frame_id").value
        ).lstrip("/")
        self._vision_frame_id = str(
            self.get_parameter("vision_frame_id").value
        ).lstrip("/")
        self._odom_frame_id = str(
            self.get_parameter("odom_frame_id").value
        ).lstrip("/")
        self._body_frame_id = str(
            self.get_parameter("body_frame_id").value
        ).lstrip("/")
        self._lidar_frame_id = str(
            self.get_parameter("lidar_frame_id").value
        ).lstrip("/")
        self._tf_root = str(self.get_parameter("tf_root").value).strip("/")
        self._map_frame_id = str(self.get_parameter("map_frame_id").value).lstrip("/")
        self._lamp_base_frame_id = str(
            self.get_parameter("lamp_base_frame_id").value
        ).lstrip("/")
        self._radiation_source_topic = str(
            self.get_parameter("radiation_source_topic").value
        )
        self._lamp_mount_xyz = self._get_vector_parameter("lamp_mount_xyz")
        self._lamp_mount_rpy = self._get_vector_parameter("lamp_mount_rpy")
        self._max_payload_bytes = int(
            self.get_parameter("max_payload_bytes").value
        )
        self._max_lidar_payload_bytes = int(
            self.get_parameter("max_lidar_payload_bytes").value
        )
        self._command_timeout = float(
            self.get_parameter("command_timeout_seconds").value
        )
        self._rgbd_publish_rate_hz = float(
            self.get_parameter("rgbd_publish_rate_hz").value
        )
        if self._max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        if self._max_lidar_payload_bytes < LIDAR_HEADER.size + 4:
            raise ValueError("max_lidar_payload_bytes is too small for one ray")
        if self._command_timeout <= 0:
            raise ValueError("command_timeout_seconds must be positive")
        if (
            not math.isfinite(self._rgbd_publish_rate_hz)
            or self._rgbd_publish_rate_hz <= 0
        ):
            raise ValueError("rgbd_publish_rate_hz must be finite and positive")
        if self._tf_root not in ("odom", "vision", "body"):
            raise ValueError("tf_root must be one of: odom, vision, body")
        if not self._map_frame_id:
            raise ValueError("map_frame_id must not be empty")
        if not self._camera_name:
            raise ValueError("camera_name must not be empty")
        if not self._lidar_frame_id:
            raise ValueError("lidar_frame_id must not be empty")
        if not self._lamp_base_frame_id:
            raise ValueError("lamp_base_frame_id must not be empty")

        camera_topic = self._spot_topic(
            str(self.get_parameter("camera_topic").value)
        )
        best_effort_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self._camera_publisher = self.create_publisher(
            CompressedImage, camera_topic, best_effort_qos
        )
        camera_prefix = self._spot_topic(f"camera/{self._camera_name}")
        self._named_camera_compressed_publisher = self.create_publisher(
            CompressedImage, f"{camera_prefix}/image/compressed", best_effort_qos
        )
        self._camera_image_publisher = self.create_publisher(
            Image, f"{camera_prefix}/image", best_effort_qos
        )
        self._camera_info_publisher = self.create_publisher(
            CameraInfo, f"{camera_prefix}/camera_info", best_effort_qos
        )
        self._camera_depth_publisher = self.create_publisher(
            Image,
            self._spot_topic(f"depth_registered/{self._camera_name}/image"),
            best_effort_qos,
        )
        self._lidar_publisher = self.create_publisher(
            PointCloud2, self._spot_topic("nav2_points_fused"), best_effort_qos
        )
        self._odom_publisher = self.create_publisher(
            Odometry, self._spot_topic("odometry"), 1
        )
        self._twist_publisher = self.create_publisher(
            TwistWithCovarianceStamped, self._spot_topic("odometry/twist"), 1
        )
        self._lease_publisher = self.create_publisher(
            Bool, self._spot_topic("status/local_lease"), 1
        )
        self._lease_held_publisher = self.create_publisher(
            Bool, self._spot_topic("lease_held"), 1
        )
        self._feedback_publisher = self.create_publisher(
            Feedback, self._spot_topic("status/feedback"), 1
        )
        self._power_publisher = self.create_publisher(
            PowerState, self._spot_topic("status/power_states"), 1
        )
        source_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._radiation_source_publisher = self.create_publisher(
            SimulatedSourceArray, self._radiation_source_topic, source_qos
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self._static_tf_broadcaster = StaticTransformBroadcaster(self)
        self._velocity_subscription = self.create_subscription(
            Twist, self._spot_topic("cmd_vel"), self._velocity_callback, 1
        )

        self._services = []
        for service_name, command in (
            ("take_lease", "claim"),
            ("release", "release"),
            ("power_on", "power_on"),
            ("power_off", "power_off"),
            ("stand", "stand"),
            ("sit", "sit"),
            ("stop", "stop"),
        ):
            self._services.append(
                self.create_service(
                    Trigger,
                    self._spot_topic(service_name),
                    self._make_command_callback(command),
                )
            )

        self._stop_event = threading.Event()
        self._rgbd_condition = threading.Condition()
        self._latest_rgbd_frame: Optional[Tuple[bytes, Any]] = None
        self._camera_client: Optional[socket.socket] = None
        self._control_client: Optional[socket.socket] = None
        self._lidar_client: Optional[socket.socket] = None
        self._control_writer = None
        self._control_send_lock = threading.Lock()
        self._control_state_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: Dict[str, Tuple[threading.Event, Optional[Dict[str, Any]]]] = {}
        self._frame_count = 0
        self._report_started: Optional[float] = None
        self._lidar_scan_count = 0
        self._lidar_report_started: Optional[float] = None
        self._lidar_mount: Optional[Tuple[float, ...]] = None
        self._camera_mount: Optional[Tuple[float, ...]] = None
        self._lidar_transform: Optional[TransformStamped] = None
        self._camera_transform: Optional[TransformStamped] = None
        self._lidar_geometry: Optional[Tuple[float, ...]] = None
        self._lidar_unit_vectors: Tuple[Tuple[float, float, float], ...] = ()

        self._camera_listener = self._create_listener(self._camera_port)
        self._control_listener = self._create_listener(self._control_port)
        self._lidar_listener = self._create_listener(self._lidar_port)
        self._camera_thread = threading.Thread(
            target=self._serve_camera,
            name="unity-camera-tcp",
            daemon=True,
        )
        self._rgbd_thread = threading.Thread(
            target=self._serve_latest_rgbd,
            name="unity-camera-rgbd",
            daemon=True,
        )
        self._control_thread = threading.Thread(
            target=self._serve_control,
            name="unity-spot-control-tcp",
            daemon=True,
        )
        self._lidar_thread = threading.Thread(
            target=self._serve_lidar,
            name="unity-lidar-tcp",
            daemon=True,
        )
        self._publish_static_transforms()
        self._camera_thread.start()
        self._rgbd_thread.start()
        self._control_thread.start()
        self._lidar_thread.start()
        self._publish_lease_state(False)

    def _publish_lease_state(self, has_lease: bool) -> None:
        message = Bool(data=has_lease)
        self._lease_publisher.publish(message)
        self._lease_held_publisher.publish(message)

    def _get_vector_parameter(self, name: str) -> Tuple[float, float, float]:
        value = self.get_parameter(name).value
        if len(value) != 3:
            raise ValueError(f"{name} must have exactly 3 values")
        return float(value[0]), float(value[1]), float(value[2])

    def _tf_root_frame_id(self) -> str:
        if self._tf_root == "odom":
            return self._odom_frame_id
        if self._tf_root == "vision":
            return self._vision_frame_id
        return self._body_frame_id

    def _publish_static_transforms(self) -> None:
        stamp = self.get_clock().now().to_msg()
        map_to_tf_root = self._make_transform(
            stamp,
            self._map_frame_id,
            self._tf_root_frame_id(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        lamp_qx, lamp_qy, lamp_qz, lamp_qw = self._quaternion_from_rpy(
            *self._lamp_mount_rpy
        )
        body_to_lamp = self._make_transform(
            stamp,
            self._body_frame_id,
            self._lamp_base_frame_id,
            self._lamp_mount_xyz[0],
            self._lamp_mount_xyz[1],
            self._lamp_mount_xyz[2],
            lamp_qx,
            lamp_qy,
            lamp_qz,
            lamp_qw,
        )
        self._base_static_transforms = [
            map_to_tf_root,
            body_to_lamp,
        ]
        self._static_tf_broadcaster.sendTransform(self._base_static_transforms)
        self.get_logger().info("Published canonical map and LAMP static TFs")

    def _publish_all_static_transforms(self) -> None:
        """Keep every sensor transform in the transient-local TF sample.

        StaticTransformBroadcaster uses a depth-one transient publisher. If
        camera and lidar updates are sent independently, the later message can
        become the only sample delivered to late-joining nodes. Publishing the
        complete set ensures every subscriber receives both sensor frames.
        """
        transforms = list(self._base_static_transforms)
        if self._camera_transform is not None:
            transforms.append(self._camera_transform)
        if self._lidar_transform is not None:
            transforms.append(self._lidar_transform)
        self._static_tf_broadcaster.sendTransform(transforms)

    @staticmethod
    def _normalize_quaternion(
        x: float, y: float, z: float, w: float
    ) -> Tuple[float, float, float, float]:
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm <= 1.0e-9:
            return 0.0, 0.0, 0.0, 1.0
        return x / norm, y / norm, z / norm, w / norm

    @staticmethod
    def _quaternion_from_rpy(
        roll: float, pitch: float, yaw: float
    ) -> Tuple[float, float, float, float]:
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return UnitySpotBridge._normalize_quaternion(
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    @staticmethod
    def _inverse_transform(transform: TransformStamped) -> TransformStamped:
        inverse = TransformStamped()
        inverse.header.stamp = transform.header.stamp
        inverse.header.frame_id = transform.child_frame_id
        inverse.child_frame_id = transform.header.frame_id

        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        tz = transform.transform.translation.z

        inverse.transform.rotation.x = -qx
        inverse.transform.rotation.y = -qy
        inverse.transform.rotation.z = -qz
        inverse.transform.rotation.w = qw

        # Inverse translation is -(R^-1 * t), using the conjugate quaternion.
        ix, iy, iz = UnitySpotBridge._rotate_vector(-qx, -qy, -qz, qw, tx, ty, tz)
        inverse.transform.translation.x = -ix
        inverse.transform.translation.y = -iy
        inverse.transform.translation.z = -iz
        return inverse

    @staticmethod
    def _rotate_vector(
        qx: float, qy: float, qz: float, qw: float, x: float, y: float, z: float
    ) -> Tuple[float, float, float]:
        # Quaternion-vector multiplication, expanded to avoid extra dependencies.
        uvx = qy * z - qz * y
        uvy = qz * x - qx * z
        uvz = qx * y - qy * x
        uuvx = qy * uvz - qz * uvy
        uuvy = qz * uvx - qx * uvz
        uuvz = qx * uvy - qy * uvx
        return (
            x + 2.0 * (qw * uvx + uuvx),
            y + 2.0 * (qw * uvy + uuvy),
            z + 2.0 * (qw * uvz + uuvz),
        )

    def _spot_topic(self, relative_name: str) -> str:
        relative_name = relative_name.strip("/")
        if not self._spot_name:
            return f"/{relative_name}"
        return f"/{self._spot_name}/{relative_name}"

    def _create_listener(self, port: int) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(1.0)
        listener.bind((self._listen_host, port))
        listener.listen(1)
        return listener

    def _serve_camera(self) -> None:
        self.get_logger().info(
            f"Listening for Unity RGB-D camera frames on "
            f"{self._listen_host}:{self._camera_port}"
        )
        try:
            while not self._stop_event.is_set():
                try:
                    client, address = self._camera_listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise

                self._camera_client = client
                client.settimeout(1.0)
                self.get_logger().info(
                    f"Unity camera connected from {address[0]}:{address[1]}"
                )
                try:
                    self._receive_frames(client)
                except (InvalidFrameError, OSError) as error:
                    self.get_logger().warning(
                        f"Closing Unity camera connection: {error}"
                    )
                finally:
                    self._camera_client = None
                    self._close_socket(client)
        finally:
            self._close_socket(self._camera_listener)

    def _receive_frames(self, client: socket.socket) -> None:
        while not self._stop_event.is_set():
            header = receive_exact(client, HEADER_SIZE)
            if header is None:
                return

            payload_size = struct.unpack("!I", header)[0]
            if payload_size == 0:
                raise InvalidFrameError("received zero-length payload")
            if payload_size > self._max_payload_bytes:
                raise InvalidFrameError(
                    f"payload size {payload_size} exceeds maximum "
                    f"{self._max_payload_bytes}"
                )

            payload = receive_exact(client, payload_size)
            if payload is None:
                raise InvalidFrameError("connection closed before camera payload")

            if payload.startswith(CAMERA_MAGIC):
                sequence, jpeg_size, depth_size = self._publish_rgbd_frame(payload)
            else:
                sequence = None
                jpeg_size = len(payload)
                depth_size = 0
                self._publish_legacy_jpeg(payload)

            self._frame_count += 1
            now = time.monotonic()
            if self._report_started is None:
                self._report_started = now
            if self._frame_count % 30 == 0:
                elapsed = now - self._report_started
                frame_intervals = 29 if self._frame_count == 30 else 30
                fps = frame_intervals / elapsed if elapsed > 0 else 0.0
                self.get_logger().info(
                    f"frames={self._frame_count} sequence={sequence} "
                    f"jpeg_bytes={jpeg_size} depth_bytes={depth_size} "
                    f"fps={fps:.1f}"
                )
                self._report_started = now

    def _publish_legacy_jpeg(self, jpeg: bytes) -> None:
        message = CompressedImage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._camera_frame_id
        message.format = "jpeg"
        message.data = jpeg
        self._camera_publisher.publish(message)
        self._named_camera_compressed_publisher.publish(message)

    def _publish_rgbd_frame(self, payload: bytes) -> Tuple[int, int, int]:
        """Publish the low-latency JPEG and replace the pending RGB-D frame."""
        frame = self._parse_rgbd_frame(payload)

        stamp = self.get_clock().now().to_msg()
        compressed = CompressedImage()
        compressed.header.stamp = stamp
        compressed.header.frame_id = self._camera_frame_id
        compressed.format = "jpeg"
        compressed.data = frame.jpeg
        self._camera_publisher.publish(compressed)
        self._named_camera_compressed_publisher.publish(compressed)

        mount_x, mount_y, mount_z, mount_qx, mount_qy, mount_qz, mount_qw = (
            frame.mount
        )
        mount_qx, mount_qy, mount_qz, mount_qw = self._normalize_quaternion(
            mount_qx, mount_qy, mount_qz, mount_qw
        )
        mount = (
            mount_x,
            mount_y,
            mount_z,
            mount_qx,
            mount_qy,
            mount_qz,
            mount_qw,
        )
        if mount != self._camera_mount:
            self._camera_transform = self._make_transform(
                stamp,
                self._body_frame_id,
                self._camera_frame_id,
                *mount,
            )
            self._publish_all_static_transforms()
            self._camera_mount = mount

        with self._rgbd_condition:
            self._latest_rgbd_frame = (payload, stamp)
            self._rgbd_condition.notify()

        return frame.sequence, len(frame.jpeg), len(frame.depth)

    def _parse_rgbd_frame(self, payload: bytes) -> RgbdFrame:
        if len(payload) < CAMERA_HEADER.size:
            raise InvalidFrameError(
                f"camera payload has {len(payload)} bytes; header requires "
                f"{CAMERA_HEADER.size}"
            )
        (
            magic,
            version,
            flags,
            sequence,
            width,
            height,
            fx,
            fy,
            cx,
            cy,
            near_clip,
            far_clip,
            mount_x,
            mount_y,
            mount_z,
            mount_qx,
            mount_qy,
            mount_qz,
            mount_qw,
            jpeg_size,
            depth_size,
        ) = CAMERA_HEADER.unpack_from(payload)
        if magic != CAMERA_MAGIC:
            raise InvalidFrameError(f"invalid camera magic {magic!r}")
        if version != CAMERA_VERSION:
            raise InvalidFrameError(f"unsupported camera version {version}")
        if flags != 0:
            raise InvalidFrameError(f"unsupported camera flags {flags}")
        if width == 0 or height == 0:
            raise InvalidFrameError("camera dimensions must be positive")
        metadata = (
            fx,
            fy,
            cx,
            cy,
            near_clip,
            far_clip,
            mount_x,
            mount_y,
            mount_z,
            mount_qx,
            mount_qy,
            mount_qz,
            mount_qw,
        )
        if not all(math.isfinite(value) for value in metadata):
            raise InvalidFrameError("camera metadata contains non-finite values")
        if fx <= 0.0 or fy <= 0.0:
            raise InvalidFrameError("camera focal lengths must be positive")
        if near_clip <= 0.0 or far_clip <= near_clip:
            raise InvalidFrameError("camera clip planes are invalid")
        expected_depth_size = width * height * 2
        if depth_size != expected_depth_size:
            raise InvalidFrameError(
                f"camera depth has {depth_size} bytes; expected "
                f"{expected_depth_size} for {width}x{height} 16UC1"
            )
        expected_size = CAMERA_HEADER.size + jpeg_size + depth_size
        if len(payload) != expected_size:
            raise InvalidFrameError(
                f"camera payload has {len(payload)} bytes; expected {expected_size}"
            )

        jpeg_start = CAMERA_HEADER.size
        jpeg_end = jpeg_start + jpeg_size
        jpeg = payload[jpeg_start:jpeg_end]
        depth = payload[jpeg_end:]
        return RgbdFrame(
            sequence=sequence,
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            near_clip=near_clip,
            far_clip=far_clip,
            mount=(
                mount_x,
                mount_y,
                mount_z,
                mount_qx,
                mount_qy,
                mount_qz,
                mount_qw,
            ),
            jpeg=jpeg,
            depth=depth,
        )

    def _serve_latest_rgbd(self) -> None:
        """Publish at most one raw RGB-D pair per interval, always the newest."""
        interval = 1.0 / self._rgbd_publish_rate_hz
        next_publish_time = time.monotonic()
        while not self._stop_event.is_set():
            with self._rgbd_condition:
                while self._latest_rgbd_frame is None and not self._stop_event.is_set():
                    self._rgbd_condition.wait(timeout=1.0)
                if self._stop_event.is_set():
                    return

                delay = next_publish_time - time.monotonic()
                if delay > 0:
                    self._rgbd_condition.wait(timeout=delay)
                    continue

                payload, stamp = self._latest_rgbd_frame
                self._latest_rgbd_frame = None

            try:
                self._publish_raw_rgbd(payload, stamp)
            except InvalidFrameError as error:
                self.get_logger().error(
                    f"Dropping invalid latest RGB-D frame: {error}"
                )
            next_publish_time = time.monotonic() + interval

    def _publish_raw_rgbd(self, payload: bytes, stamp: Any) -> None:
        frame = self._parse_rgbd_frame(payload)

        decoded = cv2.imdecode(
            np.frombuffer(frame.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if decoded is None:
            raise InvalidFrameError("camera JPEG could not be decoded")
        if decoded.shape[:2] != (frame.height, frame.width):
            raise InvalidFrameError(
                f"camera JPEG is {decoded.shape[1]}x{decoded.shape[0]}; "
                f"metadata says {frame.width}x{frame.height}"
            )

        image = Image()
        image.header.stamp = stamp
        image.header.frame_id = self._camera_frame_id
        image.height = frame.height
        image.width = frame.width
        image.encoding = "bgr8"
        image.is_bigendian = False
        image.step = frame.width * 3
        image.data = decoded.tobytes()
        self._camera_image_publisher.publish(image)

        depth_image = Image()
        depth_image.header.stamp = stamp
        depth_image.header.frame_id = self._camera_frame_id
        depth_image.height = frame.height
        depth_image.width = frame.width
        depth_image.encoding = "16UC1"
        depth_image.is_bigendian = False
        depth_image.step = frame.width * 2
        depth_image.data = frame.depth
        self._camera_depth_publisher.publish(depth_image)

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self._camera_frame_id
        info.height = frame.height
        info.width = frame.width
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [
            frame.fx,
            0.0,
            frame.cx,
            0.0,
            frame.fy,
            frame.cy,
            0.0,
            0.0,
            1.0,
        ]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [
            frame.fx,
            0.0,
            frame.cx,
            0.0,
            0.0,
            frame.fy,
            frame.cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        self._camera_info_publisher.publish(info)

    def _serve_lidar(self) -> None:
        self.get_logger().info(
            f"Listening for Unity lidar scans on "
            f"{self._listen_host}:{self._lidar_port}"
        )
        try:
            while not self._stop_event.is_set():
                try:
                    client, address = self._lidar_listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise

                self._lidar_client = client
                client.settimeout(1.0)
                self.get_logger().info(
                    f"Unity lidar connected from {address[0]}:{address[1]}"
                )
                try:
                    self._receive_lidar_scans(client)
                except (InvalidFrameError, OSError) as error:
                    self.get_logger().warning(
                        f"Closing Unity lidar connection: {error}"
                    )
                finally:
                    self._lidar_client = None
                    self._close_socket(client)
        finally:
            self._close_socket(self._lidar_listener)

    def _receive_lidar_scans(self, client: socket.socket) -> None:
        while not self._stop_event.is_set():
            frame_header = receive_exact(client, HEADER_SIZE)
            if frame_header is None:
                return

            payload_size = struct.unpack("!I", frame_header)[0]
            if payload_size < LIDAR_HEADER.size + 4:
                raise InvalidFrameError(
                    f"lidar payload size {payload_size} is too small"
                )
            if payload_size > self._max_lidar_payload_bytes:
                raise InvalidFrameError(
                    f"lidar payload size {payload_size} exceeds maximum "
                    f"{self._max_lidar_payload_bytes}"
                )

            payload = receive_exact(client, payload_size)
            if payload is None:
                raise InvalidFrameError("connection closed before lidar payload")
            self._publish_lidar_scan(payload)

    def _publish_lidar_scan(self, payload: bytes) -> None:
        (
            magic,
            version,
            flags,
            sequence,
            horizontal_count,
            vertical_count,
            horizontal_min,
            horizontal_increment,
            vertical_min,
            vertical_increment,
            range_min,
            range_max,
            mount_x,
            mount_y,
            mount_z,
            mount_qx,
            mount_qy,
            mount_qz,
            mount_qw,
        ) = LIDAR_HEADER.unpack_from(payload)
        if magic != LIDAR_MAGIC:
            raise InvalidFrameError(f"invalid lidar magic {magic!r}")
        if version != LIDAR_VERSION:
            raise InvalidFrameError(f"unsupported lidar version {version}")
        if flags != 0:
            raise InvalidFrameError(f"unsupported lidar flags {flags}")
        if horizontal_count == 0 or vertical_count == 0:
            raise InvalidFrameError("lidar dimensions must be positive")
        if not all(
            math.isfinite(value)
            for value in (
                horizontal_min,
                horizontal_increment,
                vertical_min,
                vertical_increment,
                range_min,
                range_max,
                mount_x,
                mount_y,
                mount_z,
                mount_qx,
                mount_qy,
                mount_qz,
                mount_qw,
            )
        ):
            raise InvalidFrameError("lidar metadata contains non-finite values")
        if horizontal_increment <= 0.0 or vertical_increment < 0.0:
            raise InvalidFrameError("lidar angular increments are invalid")
        if range_min <= 0.0 or range_max <= range_min:
            raise InvalidFrameError("lidar range limits are invalid")

        ray_count = horizontal_count * vertical_count
        expected_size = LIDAR_HEADER.size + ray_count * 4
        if len(payload) != expected_size:
            raise InvalidFrameError(
                f"lidar payload has {len(payload)} bytes; expected {expected_size}"
            )

        mount_qx, mount_qy, mount_qz, mount_qw = self._normalize_quaternion(
            mount_qx, mount_qy, mount_qz, mount_qw
        )
        stamp = self.get_clock().now().to_msg()
        mount = (
            mount_x,
            mount_y,
            mount_z,
            mount_qx,
            mount_qy,
            mount_qz,
            mount_qw,
        )
        if mount != self._lidar_mount:
            self._lidar_transform = self._make_transform(
                stamp,
                self._body_frame_id,
                self._lidar_frame_id,
                *mount,
            )
            self._publish_all_static_transforms()
            self._lidar_mount = mount

        geometry = (
            horizontal_count,
            vertical_count,
            horizontal_min,
            horizontal_increment,
            vertical_min,
            vertical_increment,
        )
        if geometry != self._lidar_geometry:
            unit_vectors = []
            for vertical_index in range(vertical_count):
                elevation = vertical_min + vertical_index * vertical_increment
                cos_elevation = math.cos(elevation)
                sin_elevation = math.sin(elevation)
                for horizontal_index in range(horizontal_count):
                    azimuth = horizontal_min + horizontal_index * horizontal_increment
                    unit_vectors.append(
                        (
                            cos_elevation * math.cos(azimuth),
                            cos_elevation * math.sin(azimuth),
                            sin_elevation,
                        )
                    )
            self._lidar_unit_vectors = tuple(unit_vectors)
            self._lidar_geometry = geometry

        points = bytearray(ray_count * POINT_XYZ.size)
        point_offset = 0
        valid_points = 0
        ranges = struct.iter_unpack("<f", payload[LIDAR_HEADER.size :])
        for (distance,), unit_vector in zip(ranges, self._lidar_unit_vectors):
            if not math.isfinite(distance):
                continue
            if distance < range_min or distance > range_max:
                raise InvalidFrameError(
                    f"lidar range {distance} outside [{range_min}, {range_max}]"
                )

            POINT_XYZ.pack_into(
                points,
                point_offset,
                distance * unit_vector[0],
                distance * unit_vector[1],
                distance * unit_vector[2],
            )
            point_offset += POINT_XYZ.size
            valid_points += 1

        message = PointCloud2()
        message.header.stamp = stamp
        message.header.frame_id = self._lidar_frame_id
        message.height = 1
        message.width = valid_points
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = POINT_XYZ.size
        message.row_step = valid_points * POINT_XYZ.size
        message.data = points[:point_offset]
        message.is_dense = True
        self._lidar_publisher.publish(message)

        self._lidar_scan_count += 1
        now = time.monotonic()
        if self._lidar_report_started is None:
            self._lidar_report_started = now
        if self._lidar_scan_count % 50 == 0:
            elapsed = now - self._lidar_report_started
            scan_intervals = 49 if self._lidar_scan_count == 50 else 50
            scan_rate = scan_intervals / elapsed if elapsed > 0 else 0.0
            self.get_logger().info(
                f"lidar_scans={self._lidar_scan_count} sequence={sequence} "
                f"points={valid_points}/{ray_count} rate_hz={scan_rate:.1f}"
            )
            self._lidar_report_started = now

    def _serve_control(self) -> None:
        self.get_logger().info(
            f"Listening for Unity Spot control/state on "
            f"{self._listen_host}:{self._control_port}"
        )
        try:
            while not self._stop_event.is_set():
                try:
                    client, address = self._control_listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop_event.is_set():
                        break
                    raise

                self.get_logger().info(
                    f"Unity Spot control connected from {address[0]}:{address[1]}"
                )
                reader = client.makefile("r", encoding="utf-8", newline="\n")
                writer = client.makefile("w", encoding="utf-8", newline="\n")
                with self._control_state_lock:
                    self._control_client = client
                    self._control_writer = writer
                try:
                    for line in reader:
                        if self._stop_event.is_set():
                            break
                        self._handle_control_message(line)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    self.get_logger().warning(
                        f"Closing Unity control connection: {error}"
                    )
                finally:
                    with self._control_state_lock:
                        self._control_client = None
                        self._control_writer = None
                    try:
                        reader.close()
                    except OSError:
                        pass
                    try:
                        writer.close()
                    except OSError:
                        pass
                    self._close_socket(client)
                    self._fail_pending("Unity control connection closed")
                    self._publish_lease_state(False)
        finally:
            self._close_socket(self._control_listener)

    def _handle_control_message(self, line: str) -> None:
        message = json.loads(line)
        message_type = message.get("type")
        if message_type == "pose":
            self._publish_pose(message)
        elif message_type == "radiation_sources":
            self._publish_radiation_sources(message)
        elif message_type == "response":
            request_id = str(message.get("id", ""))
            with self._pending_lock:
                pending = self._pending.get(request_id)
                if pending is not None:
                    self._pending[request_id] = (pending[0], message)
                    pending[0].set()

    def _publish_radiation_sources(self, state: Dict[str, Any]) -> None:
        raw_sources = state.get("sources")
        if not isinstance(raw_sources, list):
            raise ValueError("radiation_sources.sources must be a list")
        if len(raw_sources) > 256:
            raise ValueError("radiation source snapshot exceeds 256 sources")

        message = SimulatedSourceArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self._map_frame_id
        message.sequence = int(state.get("sequence", 0))
        message.background_rate_per_detector = float(
            state.get("background_rate_per_detector", 0.0)
        )
        if (
            not math.isfinite(message.background_rate_per_detector)
            or message.background_rate_per_detector < 0
        ):
            raise ValueError("invalid radiation background rate")

        source_ids = set()
        for raw_source in raw_sources:
            if not isinstance(raw_source, dict):
                raise ValueError("each radiation source must be an object")
            source_id = str(raw_source.get("id", "")).strip()
            isotope = str(raw_source.get("isotope", "")).strip()
            activity_bq = float(raw_source.get("activity_bq", 0.0))
            if not source_id or source_id in source_ids:
                raise ValueError(f"invalid or duplicate radiation source id: {source_id}")
            if not isotope:
                raise ValueError(f"source {source_id} has no isotope")
            if not math.isfinite(activity_bq) or activity_bq < 0:
                raise ValueError(f"source {source_id} has invalid activity")
            source_ids.add(source_id)

            source = SimulatedSource()
            source.id = source_id
            source.isotope = isotope
            source.activity_bq = activity_bq
            source.pose.position.x = float(raw_source.get("x", 0.0))
            source.pose.position.y = float(raw_source.get("y", 0.0))
            source.pose.position.z = float(raw_source.get("z", 0.0))
            source.pose.orientation.x = float(raw_source.get("qx", 0.0))
            source.pose.orientation.y = float(raw_source.get("qy", 0.0))
            source.pose.orientation.z = float(raw_source.get("qz", 0.0))
            source.pose.orientation.w = float(raw_source.get("qw", 1.0))
            pose_values = (
                source.pose.position.x,
                source.pose.position.y,
                source.pose.position.z,
                source.pose.orientation.x,
                source.pose.orientation.y,
                source.pose.orientation.z,
                source.pose.orientation.w,
            )
            if not all(math.isfinite(value) for value in pose_values):
                raise ValueError(f"source {source_id} has an invalid pose")
            message.sources.append(source)

        self._radiation_source_publisher.publish(message)

    def _state_quaternion(self, state: Dict[str, Any]) -> Tuple[float, float, float, float]:
        if "qw" in state:
            return self._normalize_quaternion(
                float(state.get("qx", 0.0)),
                float(state.get("qy", 0.0)),
                float(state.get("qz", 0.0)),
                float(state.get("qw", 1.0)),
            )

        yaw = float(state.get("yaw", 0.0))
        half_yaw = yaw * 0.5
        return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)

    def _state_vision_transform(
        self, state: Dict[str, Any]
    ) -> Tuple[float, float, float, float, float, float, float]:
        return (
            float(state.get("vision_x", 0.0)),
            float(state.get("vision_y", 0.0)),
            float(state.get("vision_z", 0.0)),
            *self._normalize_quaternion(
                float(state.get("vision_qx", 0.0)),
                float(state.get("vision_qy", 0.0)),
                float(state.get("vision_qz", 0.0)),
                float(state.get("vision_qw", 1.0)),
            ),
        )

    @staticmethod
    def _make_transform(
        stamp,
        parent_frame_id: str,
        child_frame_id: str,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = parent_frame_id
        transform.child_frame_id = child_frame_id
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        return transform

    def _publish_pose(self, state: Dict[str, Any]) -> None:
        stamp = self.get_clock().now().to_msg()
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        z = float(state.get("z", 0.0))
        qx, qy, qz, qw = self._state_quaternion(state)
        (
            vision_x,
            vision_y,
            vision_z,
            vision_qx,
            vision_qy,
            vision_qz,
            vision_qw,
        ) = self._state_vision_transform(state)
        vx = float(state.get("vx", 0.0))
        vy = float(state.get("vy", 0.0))
        wz = float(state.get("wz", 0.0))
        has_lease = bool(state.get("has_lease", False))
        powered_on = bool(state.get("powered_on", False))
        standing = bool(state.get("standing", False))
        moving = abs(vx) > 0.01 or abs(vy) > 0.01 or abs(wz) > 0.01
        self._publish_lease_state(has_lease)
        feedback = Feedback()
        feedback.standing = standing
        feedback.sitting = not standing
        feedback.moving = moving
        feedback.serial_number = "unity_spot"
        feedback.species = "spot"
        feedback.version = "unity"
        feedback.nickname = "Unity Spot"
        feedback.computer_serial_number = "unity"
        self._feedback_publisher.publish(feedback)

        power = PowerState()
        power.header.stamp = stamp
        power.header.frame_id = self._body_frame_id
        power.motor_power_state = (
            PowerState.STATE_ON if powered_on else PowerState.STATE_OFF
        )
        power.shore_power_state = PowerState.STATE_UNKNOWN_SHORE_POWER
        power.locomotion_charge_percentage = 100.0
        self._power_publisher.publish(power)

        odometry = Odometry()
        odometry.header.stamp = stamp
        odometry.header.frame_id = self._odom_frame_id
        odometry.child_frame_id = self._body_frame_id
        odometry.pose.pose.position.x = x
        odometry.pose.pose.position.y = y
        odometry.pose.pose.position.z = z
        odometry.pose.pose.orientation.x = qx
        odometry.pose.pose.orientation.y = qy
        odometry.pose.pose.orientation.z = qz
        odometry.pose.pose.orientation.w = qw
        odometry.twist.twist.linear.x = vx
        odometry.twist.twist.linear.y = vy
        odometry.twist.twist.angular.z = wz
        self._odom_publisher.publish(odometry)

        twist = TwistWithCovarianceStamped()
        twist.header.stamp = stamp
        twist.header.frame_id = self._body_frame_id
        twist.twist = odometry.twist
        self._twist_publisher.publish(twist)

        odom_to_vision = self._make_transform(
            stamp,
            self._odom_frame_id,
            self._vision_frame_id,
            vision_x,
            vision_y,
            vision_z,
            vision_qx,
            vision_qy,
            vision_qz,
            vision_qw,
        )
        odom_to_body = self._make_transform(
            stamp,
            self._odom_frame_id,
            self._body_frame_id,
            x,
            y,
            z,
            qx,
            qy,
            qz,
            qw,
        )
        if self._tf_root == "odom":
            self._tf_broadcaster.sendTransform([odom_to_vision, odom_to_body])
        elif self._tf_root == "vision":
            self._tf_broadcaster.sendTransform(
                [self._inverse_transform(odom_to_vision), odom_to_body]
            )
        else:
            self._tf_broadcaster.sendTransform(
                [self._inverse_transform(odom_to_body), odom_to_vision]
            )

    def _velocity_callback(self, message: Twist) -> None:
        self._send_json(
            {
                "type": "velocity",
                "vx": message.linear.x,
                "vy": message.linear.y,
                "wz": message.angular.z,
            }
        )

    def _make_command_callback(self, command: str):
        def callback(_request: Trigger.Request, response: Trigger.Response):
            response.success, response.message = self._send_command(command)
            if response.success and command in ("claim", "release"):
                self._publish_lease_state(command == "claim")
            return response

        return callback

    def _send_command(self, command: str) -> Tuple[bool, str]:
        request_id = uuid.uuid4().hex
        event = threading.Event()
        with self._pending_lock:
            self._pending[request_id] = (event, None)

        if not self._send_json(
            {"type": "command", "id": request_id, "command": command}
        ):
            with self._pending_lock:
                self._pending.pop(request_id, None)
            return False, "Unity control bridge is not connected"

        if not event.wait(self._command_timeout):
            with self._pending_lock:
                self._pending.pop(request_id, None)
            return False, f"Timed out waiting for Unity to execute {command}"

        with self._pending_lock:
            pending = self._pending.pop(request_id, None)
        result = pending[1] if pending is not None else None
        if result is None:
            return False, "Unity control connection closed"
        return bool(result.get("success", False)), str(result.get("message", ""))

    def _send_json(self, message: Dict[str, Any]) -> bool:
        encoded = json.dumps(message, separators=(",", ":"))
        with self._control_send_lock:
            with self._control_state_lock:
                writer = self._control_writer
            if writer is None:
                return False
            try:
                writer.write(encoded + "\n")
                writer.flush()
                return True
            except (OSError, ValueError):
                return False

    def _fail_pending(self, reason: str) -> None:
        with self._pending_lock:
            for request_id, (event, _response) in tuple(self._pending.items()):
                self._pending[request_id] = (
                    event,
                    {"success": False, "message": reason},
                )
                event.set()

    @staticmethod
    def _close_socket(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            sock.close()
        except OSError:
            pass

    def stop(self) -> None:
        self._stop_event.set()
        with self._rgbd_condition:
            self._rgbd_condition.notify_all()
        for sock in (
            self._camera_client,
            self._control_client,
            self._lidar_client,
            self._camera_listener,
            self._control_listener,
            self._lidar_listener,
        ):
            if sock is not None:
                self._close_socket(sock)
        self._fail_pending("Bridge is shutting down")
        for thread in (
            self._camera_thread,
            self._rgbd_thread,
            self._control_thread,
            self._lidar_thread,
        ):
            if thread.is_alive():
                thread.join(timeout=2.0)


# Preserve the old import name for downstream users of the camera-only bridge.
UnityCameraBridge = UnitySpotBridge


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UnitySpotBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
