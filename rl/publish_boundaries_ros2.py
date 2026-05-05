"""
Small ROS2 node to eyeball saved boundaries in RViz. Loads the outer_line.npy /
inner_line.npy the mapping worker writes, smooths + resamples them to even
spacing, and publishes a MarkerArray.

Run it anywhere with ROS2 and a checkout of this repo (it imports
control/boundary.py), e.g. the autodrive bridge container with the repo mounted:
    python3 publish_boundaries_ros2.py \
        --outer outer_line.npy --inner inner_line.npy \
        --spacing 0.1 --rate 1.0

Publishes /pilot/saved_boundary in frame "world": red = outer, blue = inner.
"""

import argparse
import os
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# same smoothing/resampling as training and the car (control/boundary.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from control.boundary import resample_uniform, smooth_boundary  # noqa: E402


class BoundaryPublisher(Node):

    def __init__(self, outer: np.ndarray, inner: np.ndarray, rate_hz: float = 1.0):
        super().__init__('boundary_publisher')
        self._outer = outer
        self._inner = inner
        self._pub = self.create_publisher(
            MarkerArray,
            '/pilot/saved_boundary',
            10,
        )
        self._timer = self.create_timer(1.0 / rate_hz, self._publish)
        self.get_logger().info(
            f"Publishing resampled boundaries: {len(outer)} outer pts, {len(inner)} inner pts"
        )

    def _publish(self):
        now = self.get_clock().now().to_msg()
        ma = MarkerArray()

        for pts, mid, (r, g, b) in [
            (self._outer, 0, (1.0, 0.0, 0.0)),  # red
            (self._inner, 1, (0.0, 0.0, 1.0)),  # blue
        ]:
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = 'world'
            m.ns = 'boundary'
            m.id = mid
            m.type = Marker.POINTS
            m.action = Marker.ADD
            m.scale.x = 0.05  # point diameter
            m.scale.y = 0.05
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            for pt in pts:
                p = Point()
                p.x = float(pt[0]); p.y = float(pt[1]); p.z = 0.0
                m.points.append(p)
            ma.markers.append(m)

        self._pub.publish(ma)


def main():
    parser = argparse.ArgumentParser(description="Publish saved track boundaries to RViz.")
    parser.add_argument('--outer', default='outer_line.npy', help='outer boundary .npy (N, >=2)')
    parser.add_argument('--inner', default='inner_line.npy', help='inner boundary .npy (M, >=2)')
    parser.add_argument('--spacing', type=float, default=0.25,
                        help='Uniform arc-length spacing in metres (default 0.25)')
    parser.add_argument('--smooth', type=int, default=15,
                        help='Moving-average window size for smoothing (default 15, 0=off)')
    parser.add_argument('--rate', type=float, default=1.0,
                        help='Publish rate in Hz (default 1.0)')
    args = parser.parse_args()

    outer_raw = np.load(args.outer)[:, :2]
    inner_raw = np.load(args.inner)[:, :2]

    if args.smooth > 0:
        outer_raw = smooth_boundary(outer_raw, window=args.smooth)
        inner_raw = smooth_boundary(inner_raw, window=args.smooth)

    outer = resample_uniform(outer_raw, args.spacing)
    inner = resample_uniform(inner_raw, args.spacing)

    print(f"smooth={args.smooth}  spacing={args.spacing}m  "
          f"outer→{len(outer)} pts  inner→{len(inner)} pts")

    rclpy.init()
    node = BoundaryPublisher(outer, inner, rate_hz=args.rate)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
