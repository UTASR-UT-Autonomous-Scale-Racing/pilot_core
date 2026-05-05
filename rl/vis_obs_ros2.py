"""
Shows what TrackEnv is actually observing, in RViz.

TrackEnv pushes the world-frame boundary points of each observation over a
ZMQ PUSH socket when it has a vis_zmq_port (off by default; start training
with `python rl/train.py --vis-port 5558`). This node pulls them and
republishes as a MarkerArray on /pilot/obs_points. Run it on the same host as
rl/train.py from anywhere with ROS2, e.g. the autodrive_devkit container:

    python3 rl/vis_obs_ros2.py [--port 5558] [--frame-id world]

Markers: yellow strip "obs_outer" (from the message's "inner" points - see
the note in _publish_markers), cyan strip "obs_inner" (from "outer"), and a
white sphere "obs_pos" at the car.
"""

import argparse
import json

import numpy as np
import zmq
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import StaticTransformBroadcaster


def _make_line_marker(frame_id: str, ns: str, mid: int,
                      r: float, g: float, b: float,
                      width: float = 0.05) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = mid
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = width
    m.color.r = r
    m.color.g = g
    m.color.b = b
    m.color.a = 1.0
    m.pose.orientation.w = 1.0  # all-zero quaternion is invalid, RViz complains
    return m


class ObsVisNode(Node):

    def __init__(self, zmq_port: int, frame_id: str):
        super().__init__("obs_vis")
        self._frame_id = frame_id

        self._pub = self.create_publisher(MarkerArray, "/pilot/obs_points", 10)

        ctx = zmq.Context.instance()
        self._sock = ctx.socket(zmq.PULL)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.connect(f"tcp://localhost:{zmq_port}")

        # static TF, otherwise RViz doesn't know the 'world' frame exists
        self._tf_broadcaster = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "world"
        tf.child_frame_id = "world_origin"
        tf.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf)

        # poll on a timer with NOBLOCK so spin never stalls
        self.create_timer(1.0 / 30.0, self._poll)
        self.get_logger().info(
            f"Subscribed to tcp://localhost:{zmq_port} → /pilot/obs_points"
        )

    def _poll(self):
        try:
            raw = self._sock.recv_string(flags=zmq.NOBLOCK)
        except zmq.Again:
            return

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        self.get_logger().info("received obs frame", throttle_duration_sec=2.0)

        try:
            self._publish_markers(data)
        except Exception as e:
            self.get_logger().error(f"publish failed: {e}")

    def _publish_markers(self, data: dict):
        pos = np.array(data["pos"], dtype=np.float32)
        outer = np.array(data["outer"], dtype=np.float32)
        inner = np.array(data["inner"], dtype=np.float32)

        now = self.get_clock().now().to_msg()
        ma = MarkerArray()

        # NB: swapped on purpose. The saved .npy files name the walls the other
        # way round, so "inner" points go into the yellow obs_outer marker
        mo = _make_line_marker(self._frame_id, "obs_outer", 0, 1.0, 1.0, 0.0)
        mo.header.stamp = now
        for pt in inner:
            p = Point()
            p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.0
            mo.points.append(p)
        ma.markers.append(mo)

        # and "outer" -> cyan obs_inner
        mi = _make_line_marker(self._frame_id, "obs_inner", 1, 0.0, 1.0, 1.0)
        mi.header.stamp = now
        for pt in outer:
            p = Point()
            p.x, p.y, p.z = float(pt[0]), float(pt[1]), 0.0
            mi.points.append(p)
        ma.markers.append(mi)

        # car position
        mp = Marker()
        mp.header.stamp = now
        mp.header.frame_id = self._frame_id
        mp.ns = "obs_pos"
        mp.id = 2
        mp.type = Marker.SPHERE
        mp.action = Marker.ADD
        mp.scale.x = mp.scale.y = mp.scale.z = 0.25
        mp.color.r = mp.color.g = mp.color.b = mp.color.a = 1.0
        mp.pose.position.x = float(pos[0])
        mp.pose.position.y = float(pos[1])
        mp.pose.position.z = 0.0
        mp.pose.orientation.w = 1.0
        ma.markers.append(mp)

        self._pub.publish(ma)


def main():
    parser = argparse.ArgumentParser(description="Visualise TrackEnv observations in RViz.")
    parser.add_argument("--port", type=int, default=5558,
                        help="ZMQ PULL port (must match vis_zmq_port in TrackEnv)")
    parser.add_argument("--frame-id", default="world",
                        help="RViz fixed frame (default: world)")
    args = parser.parse_args()

    rclpy.init()
    node = ObsVisNode(zmq_port=args.port, frame_id=args.frame_id)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
