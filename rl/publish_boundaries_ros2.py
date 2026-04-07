"""
Standalone ROS2 node that loads saved boundary .npy files, resamples them
to uniform arc-length spacing, and publishes them as a MarkerArray for rviz.

Usage (inside bridge container, with boundaries copied in):
    python3 publish_boundaries_ros2.py \
        --outer outer_line.npy --inner inner_line.npy \
        --spacing 0.1 --rate 1.0

Topic: /autodrive/roboracer_1/stack/boundary  (same as MappingWorker live output)
Red  = outer wall
Blue = inner wall
"""

import argparse
import sys
import os

import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def resample_uniform(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) < 2:
        return points.astype(np.float32)
    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = arc[-1]
    if total_len == 0.0:
        return points[:1].astype(np.float32)
    targets = np.arange(0.0, total_len, spacing)
    new_x = np.interp(targets, arc, points[:, 0])
    new_y = np.interp(targets, arc, points[:, 1])
    return np.stack([new_x, new_y], axis=1).astype(np.float32)


def smooth_boundary(points: np.ndarray, window: int = 15) -> np.ndarray:
    """Moving-average smooth to remove zigzag from clustered LIDAR wall points."""
    if len(points) < window:
        return points.astype(np.float32)
    kernel = np.ones(window) / window
    # Pad with edge values to avoid shrinking the array
    px = np.convolve(points[:, 0], kernel, mode='same')
    py = np.convolve(points[:, 1], kernel, mode='same')
    return np.stack([px, py], axis=1).astype(np.float32)


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
            (self._outer, 0, (1.0, 0.0, 0.0)),  # red = outer
            (self._inner, 1, (0.0, 0.0, 1.0)),  # blue = inner
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--outer', default='outer_line.npy')
    parser.add_argument('--inner', default='inner_line.npy')
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
