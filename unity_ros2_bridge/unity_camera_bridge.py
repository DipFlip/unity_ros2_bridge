import json
import math
import socket
import struct
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped, Twist, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from spot_msgs.msg import Feedback, PowerState
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster


HEADER_SIZE = 4


class InvalidFrameError(Exception):
    pass


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
        self.declare_parameter("spot_name", "spot")
        self.declare_parameter("camera_topic", "camera/image/compressed")
        self.declare_parameter("camera_frame_id", "spot/camera_optical_frame")
        self.declare_parameter("vision_frame_id", "spot/vision")
        self.declare_parameter("odom_frame_id", "spot/odom")
        self.declare_parameter("body_frame_id", "spot/body")
        self.declare_parameter("max_payload_bytes", 5 * 1024 * 1024)
        self.declare_parameter("command_timeout_seconds", 2.0)

        self._listen_host = str(self.get_parameter("listen_host").value)
        self._camera_port = int(self.get_parameter("camera_port").value)
        self._control_port = int(self.get_parameter("control_port").value)
        self._spot_name = str(self.get_parameter("spot_name").value).strip("/")
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
        self._max_payload_bytes = int(
            self.get_parameter("max_payload_bytes").value
        )
        self._command_timeout = float(
            self.get_parameter("command_timeout_seconds").value
        )
        if self._max_payload_bytes <= 0:
            raise ValueError("max_payload_bytes must be positive")
        if self._command_timeout <= 0:
            raise ValueError("command_timeout_seconds must be positive")

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
        self._odom_publisher = self.create_publisher(
            Odometry, self._spot_topic("odometry"), 1
        )
        self._twist_publisher = self.create_publisher(
            TwistWithCovarianceStamped, self._spot_topic("odometry/twist"), 1
        )
        self._lease_publisher = self.create_publisher(
            Bool, self._spot_topic("status/local_lease"), 1
        )
        self._feedback_publisher = self.create_publisher(
            Feedback, self._spot_topic("status/feedback"), 1
        )
        self._power_publisher = self.create_publisher(
            PowerState, self._spot_topic("status/power_states"), 1
        )
        self._tf_broadcaster = TransformBroadcaster(self)
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
        self._camera_client: Optional[socket.socket] = None
        self._control_client: Optional[socket.socket] = None
        self._control_writer = None
        self._control_send_lock = threading.Lock()
        self._control_state_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending: Dict[str, Tuple[threading.Event, Optional[Dict[str, Any]]]] = {}
        self._frame_count = 0
        self._report_started: Optional[float] = None

        self._camera_listener = self._create_listener(self._camera_port)
        self._control_listener = self._create_listener(self._control_port)
        self._camera_thread = threading.Thread(
            target=self._serve_camera,
            name="unity-camera-tcp",
            daemon=True,
        )
        self._control_thread = threading.Thread(
            target=self._serve_control,
            name="unity-spot-control-tcp",
            daemon=True,
        )
        self._camera_thread.start()
        self._control_thread.start()
        self._lease_publisher.publish(Bool(data=False))

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
            f"Listening for Unity JPEG frames on "
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
                raise InvalidFrameError("connection closed before JPEG payload")

            message = CompressedImage()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self._camera_frame_id
            message.format = "jpeg"
            message.data = payload
            self._camera_publisher.publish(message)

            self._frame_count += 1
            now = time.monotonic()
            if self._report_started is None:
                self._report_started = now
            if self._frame_count % 30 == 0:
                elapsed = now - self._report_started
                frame_intervals = 29 if self._frame_count == 30 else 30
                fps = frame_intervals / elapsed if elapsed > 0 else 0.0
                self.get_logger().info(
                    f"frames={self._frame_count} payload_bytes={payload_size} "
                    f"fps={fps:.1f}"
                )
                self._report_started = now

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
                    self._lease_publisher.publish(Bool(data=False))
        finally:
            self._close_socket(self._control_listener)

    def _handle_control_message(self, line: str) -> None:
        message = json.loads(line)
        message_type = message.get("type")
        if message_type == "pose":
            self._publish_pose(message)
        elif message_type == "response":
            request_id = str(message.get("id", ""))
            with self._pending_lock:
                pending = self._pending.get(request_id)
                if pending is not None:
                    self._pending[request_id] = (pending[0], message)
                    pending[0].set()

    def _publish_pose(self, state: Dict[str, Any]) -> None:
        stamp = self.get_clock().now().to_msg()
        x = float(state.get("x", 0.0))
        y = float(state.get("y", 0.0))
        z = float(state.get("z", 0.0))
        yaw = float(state.get("yaw", 0.0))
        vx = float(state.get("vx", 0.0))
        vy = float(state.get("vy", 0.0))
        wz = float(state.get("wz", 0.0))
        has_lease = bool(state.get("has_lease", False))
        powered_on = bool(state.get("powered_on", False))
        standing = bool(state.get("standing", False))
        moving = abs(vx) > 0.01 or abs(vy) > 0.01 or abs(wz) > 0.01
        self._lease_publisher.publish(
            Bool(data=has_lease)
        )
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

        half_yaw = yaw * 0.5
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)

        odometry = Odometry()
        odometry.header.stamp = stamp
        odometry.header.frame_id = self._odom_frame_id
        odometry.child_frame_id = self._body_frame_id
        odometry.pose.pose.position.x = x
        odometry.pose.pose.position.y = y
        odometry.pose.pose.position.z = z
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

        vision_to_odom = TransformStamped()
        vision_to_odom.header.stamp = stamp
        vision_to_odom.header.frame_id = self._vision_frame_id
        vision_to_odom.child_frame_id = self._odom_frame_id
        vision_to_odom.transform.rotation.w = 1.0

        odom_to_body = TransformStamped()
        odom_to_body.header.stamp = stamp
        odom_to_body.header.frame_id = self._odom_frame_id
        odom_to_body.child_frame_id = self._body_frame_id
        odom_to_body.transform.translation.x = x
        odom_to_body.transform.translation.y = y
        odom_to_body.transform.translation.z = z
        odom_to_body.transform.rotation.z = qz
        odom_to_body.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform([vision_to_odom, odom_to_body])

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
                self._lease_publisher.publish(Bool(data=command == "claim"))
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
        for sock in (
            self._camera_client,
            self._control_client,
            self._camera_listener,
            self._control_listener,
        ):
            if sock is not None:
                self._close_socket(sock)
        self._fail_pending("Bridge is shutting down")
        for thread in (self._camera_thread, self._control_thread):
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
