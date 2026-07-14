# Unity ROS 2 Spot bridge

`unity_spot_bridge` exposes the Unity robot with the ROS interface used by the
Spot driver in this workspace.

Run the node before entering Play mode in Unity:

```bash
ros2 run unity_ros2_bridge unity_spot_bridge
```

The bridge listens for the existing JPEG stream on TCP port 50051 and for
bidirectional command/state messages on port 50052. Defaults are:

- Services: `/spot/take_lease`, `/spot/release`, `/spot/power_on`,
  `/spot/power_off`, `/spot/stand`, `/spot/sit`, and `/spot/stop`.
- Velocity input: `/spot/cmd_vel` (`geometry_msgs/msg/Twist`).
- State: `/spot/odometry`, `/spot/odometry/twist`,
  `/spot/status/local_lease`, `/spot/status/feedback`, and
  `/spot/status/power_states`.
- TF: `spot/vision -> spot/odom -> spot/body`. Unity's starting robot pose is
  the identity of both world frames.
- Camera: `/spot/camera/image/compressed`.

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
