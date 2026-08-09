# Unity ROS 2 Spot bridge

`unity_spot_bridge` exposes the Unity robot with the ROS interface used by the
Spot driver in this workspace.

Run the node before entering Play mode in Unity:

```bash
ros2 run unity_ros2_bridge unity_spot_bridge
```

The bridge listens for the existing JPEG stream on TCP port 50051, for
bidirectional command/state messages on port 50052, and for packed synthetic
lidar scans on port 50053. Defaults are:

- Services: `/spot/take_lease`, `/spot/release`, `/spot/power_on`,
  `/spot/power_off`, `/spot/stand`, `/spot/sit`, and `/spot/stop`.
- Velocity input: `/spot/cmd_vel` (`geometry_msgs/msg/Twist`).
- State: `/spot/odometry`, `/spot/odometry/twist`,
  `/spot/status/local_lease`, `/spot/status/feedback`, and
  `/spot/status/power_states`.
- TF: `map -> spot/odom -> spot/body`, with fixed `spot/body -> lamp_base_link`
  and `spot/body -> spot/lidar` mounts, plus `spot/odom -> spot/vision`.
  Unity's perfect global pose takes the role normally provided by localization's
  `map` transform. The simulation does not publish the deployment-specific
  `map_ground` or `spot/odom_ground` convenience frames: `/map`, Nav2 goals,
  plans, and UI poses all use canonical `map` coordinates, while the local
  costmap uses `spot/odom`. `lamp_base_link` is fixed to Spot's body with the
  standard Spot LAMP mount (`xyz=[0,0,0.25]`, `rpy=[pi,0,0]`). Set `tf_root`
  to `vision` or `body` to invert the Spot subtree the same way the driver does.
- Camera: `/spot/camera/image/compressed`.
- Lidar: `/spot/nav2_points_fused` (`sensor_msgs/msg/PointCloud2`, best effort,
  `spot/lidar` frame). Unity sends packed ranges and the `Lidar` GameObject's
  body-relative pose; the bridge reconstructs XYZ points and publishes the
  corresponding TF without JSON or base64 overhead.
- Occupancy map: `unity_lidar_mapper` ray-traces the simulated point cloud using
  Unity's perfect `map -> spot/lidar` transform and publishes a persistent
  `nav_msgs/msg/OccupancyGrid` on `/map`. This provides Cartographer-compatible
  map output without running a second localization estimate in the simulation.
- Radiation sources: every enabled Unity `RadiationSource` component is sent as
  part of a complete scene snapshot and published on
  `/simulation/radiation_sources` as
  `mfdf_ros2_msgs/msg/SimulatedSourceArray`. Source poses are in `map`, sources
  may move, and the snapshot also carries the scene-wide background count rate.
  The ROS synthetic forward projector applies isotope gamma lines, branching
  ratios, detector geometry, angular response, attenuation, counting noise, and
  energy response before publishing `/interaction_data_synth`.

`take_lease` maps to Unity's internal `claim` command so the ROS API matches
the real Spot driver while the Unity scene can keep its local naming.

Movement commands are accepted by Unity only while the simulated robot has a
lease, is powered on, and is standing. Velocity commands expire after 0.35 s,
matching the short-duration command behavior of the hardware driver.

The Unity scene also exposes the same lease, power, sit, and stand state through
its on-screen control panel. When no fresh ROS velocity command is active,
`W/S` drives forward/reverse, `A/D` controls yaw, and `Q/E` strafes left/right.
Keyboard movement has the same lease, power, and standing prerequisites as
`/spot/cmd_vel`.
