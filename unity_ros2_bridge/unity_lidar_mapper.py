import math
import struct
import threading
from typing import Dict, Tuple

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

from unity_ros2_bridge.occupancy_grid import RaytracedOccupancyGrid


class UnityLidarMapper(Node):
    """Build a persistent 2D occupancy grid using Unity lidar and perfect TF."""

    def __init__(self) -> None:
        super().__init__("unity_lidar_mapper")
        self.declare_parameter("input_topic", "/spot/nav2_points_fused")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("resolution", 0.1)
        self.declare_parameter("width", 1000)
        self.declare_parameter("height", 1000)
        self.declare_parameter("origin_x", -50.0)
        self.declare_parameter("origin_y", -50.0)
        self.declare_parameter("min_obstacle_height", 0.15)
        self.declare_parameter("max_obstacle_height", 2.0)
        self.declare_parameter("min_range", 0.3)
        self.declare_parameter("max_range", 20.0)
        self.declare_parameter("angular_resolution_degrees", 1.5)
        self.declare_parameter("publish_period_seconds", 1.0)
        self.declare_parameter("publish_padding_meters", 2.0)

        self._map_frame = str(self.get_parameter("map_frame").value).lstrip("/")
        resolution = float(self.get_parameter("resolution").value)
        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        self._min_obstacle_height = float(
            self.get_parameter("min_obstacle_height").value
        )
        self._max_obstacle_height = float(
            self.get_parameter("max_obstacle_height").value
        )
        self._min_range = float(self.get_parameter("min_range").value)
        self._max_range = float(self.get_parameter("max_range").value)
        angular_resolution = math.radians(
            float(self.get_parameter("angular_resolution_degrees").value)
        )
        publish_period = float(self.get_parameter("publish_period_seconds").value)
        padding_meters = float(self.get_parameter("publish_padding_meters").value)
        if not self._map_frame:
            raise ValueError("map_frame must not be empty")
        if self._min_obstacle_height >= self._max_obstacle_height:
            raise ValueError("obstacle height range is invalid")
        if self._min_range < 0.0 or self._min_range >= self._max_range:
            raise ValueError("lidar range is invalid")
        if angular_resolution <= 0.0 or publish_period <= 0.0:
            raise ValueError("angular resolution and publish period must be positive")

        self._angular_resolution = angular_resolution
        self._padding_cells = math.ceil(padding_meters / resolution)
        self._grid = RaytracedOccupancyGrid(
            resolution=resolution,
            width=width,
            height=height,
            origin_x=float(self.get_parameter("origin_x").value),
            origin_y=float(self.get_parameter("origin_y").value),
        )
        self._grid_lock = threading.Lock()
        self._scan_count = 0
        self._last_reported_scan_count = 0
        self._last_transform_warning_ns = 0

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._publisher = self.create_publisher(
            OccupancyGrid,
            str(self.get_parameter("map_topic").value),
            map_qos,
        )
        self._subscription = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_topic").value),
            self._cloud_callback,
            cloud_qos,
        )
        self._timer = self.create_timer(publish_period, self._publish_map)
        self.get_logger().info(
            f"Mapping lidar into {width}x{height} cells at {resolution:.3f} m/cell"
        )

    def _cloud_callback(self, message: PointCloud2) -> None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self._map_frame,
                message.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.1),
            )
        except TransformException as error:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_transform_warning_ns > 5_000_000_000:
                self.get_logger().warning(f"Waiting for lidar TF: {error}")
                self._last_transform_warning_ns = now_ns
            return

        offsets = {field.name: field.offset for field in message.fields}
        if not all(name in offsets for name in ("x", "y", "z")):
            self.get_logger().error("Point cloud must contain x, y, and z fields")
            return
        if any(
            field.datatype != PointField.FLOAT32
            for field in message.fields
            if field.name in ("x", "y", "z")
        ):
            self.get_logger().error("Point cloud XYZ fields must be FLOAT32")
            return

        endian = ">" if message.is_bigendian else "<"
        unpack_float = struct.Struct(f"{endian}f").unpack_from
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        bins: Dict[int, Tuple[float, float, float]] = {}
        data = message.data
        point_count = message.width * message.height
        for point_index in range(point_count):
            offset = point_index * message.point_step
            x = unpack_float(data, offset + offsets["x"])[0]
            y = unpack_float(data, offset + offsets["y"])[0]
            z = unpack_float(data, offset + offsets["z"])[0]
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            rotated_x, rotated_y, rotated_z = self._rotate_vector(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
                x,
                y,
                z,
            )
            endpoint_z = translation.z + rotated_z
            if not self._min_obstacle_height <= endpoint_z <= self._max_obstacle_height:
                continue
            distance_squared = rotated_x * rotated_x + rotated_y * rotated_y
            if not self._min_range * self._min_range <= distance_squared:
                continue
            if distance_squared > self._max_range * self._max_range:
                continue
            bin_index = round(
                math.atan2(rotated_y, rotated_x) / self._angular_resolution
            )
            previous = bins.get(bin_index)
            if previous is None or distance_squared < previous[0]:
                bins[bin_index] = (
                    distance_squared,
                    translation.x + rotated_x,
                    translation.y + rotated_y,
                )

        if not bins:
            return
        with self._grid_lock:
            for _, hit_x, hit_y in bins.values():
                self._grid.update_ray(
                    translation.x,
                    translation.y,
                    hit_x,
                    hit_y,
                )
        self._scan_count += 1

    def _publish_map(self) -> None:
        with self._grid_lock:
            if not self._grid.has_observations:
                return
            origin_x, origin_y, width, height, data = self._grid.cropped_data(
                self._padding_cells
            )

        stamp = self.get_clock().now().to_msg()
        message = OccupancyGrid()
        message.header.stamp = stamp
        message.header.frame_id = self._map_frame
        message.info.map_load_time = stamp
        message.info.resolution = self._grid.resolution
        message.info.width = width
        message.info.height = height
        message.info.origin.position.x = origin_x
        message.info.origin.position.y = origin_y
        message.info.origin.orientation.w = 1.0
        message.data = data
        self._publisher.publish(message)
        if self._scan_count - self._last_reported_scan_count >= 50:
            self.get_logger().info(
                f"Published /map from {self._scan_count} scans ({width}x{height} cells)"
            )
            self._last_reported_scan_count = self._scan_count

    @staticmethod
    def _rotate_vector(
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[float, float, float]:
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UnityLidarMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
