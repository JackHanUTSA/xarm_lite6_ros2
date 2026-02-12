#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None


class OneShotImageSaver(Node):
    def __init__(self, topic: str, out_path: Path, timeout_sec: float = 5.0):
        super().__init__('one_shot_image_saver')
        self.out_path = out_path
        self.timeout_sec = timeout_sec
        self._got = False
        self._bridge = CvBridge() if CvBridge is not None else None

        qos = rclpy.qos.QoSProfile(depth=1)
        qos.reliability = rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT
        qos.durability = rclpy.qos.QoSDurabilityPolicy.VOLATILE
        self.create_subscription(Image, topic, self.cb, qos)

    def cb(self, msg: Image):
        if self._got:
            return
        try:
            if self._bridge is None:
                enc = msg.encoding.lower()
                if enc not in ('rgb8', 'bgr8'):
                    raise RuntimeError(f'cv_bridge missing and unsupported encoding {msg.encoding}')
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                if enc == 'rgb8':
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(self.out_path), img)
            if not ok:
                raise RuntimeError('cv2.imwrite returned False')
            self.get_logger().info(f'WROTE {self.out_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save image: {e}')
        finally:
            self._got = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topic', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--timeout', type=float, default=5.0)
    args = ap.parse_args()

    rclpy.init()
    node = OneShotImageSaver(args.topic, Path(args.out), args.timeout)

    start = node.get_clock().now()
    while rclpy.ok() and not node._got:
        rclpy.spin_once(node, timeout_sec=0.1)
        if (node.get_clock().now() - start).nanoseconds / 1e9 > args.timeout:
            node.get_logger().error('Timeout waiting for image')
            break

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
