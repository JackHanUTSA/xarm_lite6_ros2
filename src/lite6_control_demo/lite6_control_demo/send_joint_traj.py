import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class SendJointTraj(Node):
    def __init__(self):
        super().__init__('send_joint_traj')

        self.declare_parameter('topic', '/xarm6_traj_controller/joint_trajectory')
        self.declare_parameter('joint_names', [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'
        ])
        self.declare_parameter('delta_joint1', 0.10)  # radians
        self.declare_parameter('move_time', 8.0)      # seconds

        topic = self.get_parameter('topic').value
        self.joint_names = list(self.get_parameter('joint_names').value)
        self.delta = float(self.get_parameter('delta_joint1').value)
        self.move_time = float(self.get_parameter('move_time').value)

        self.pub = self.create_publisher(JointTrajectory, topic, 10)
        self.get_logger().info(f'Publishing to {topic}')
        self.get_logger().info(f'Joint names: {self.joint_names}')
        self.get_logger().info(f'Delta joint1: {self.delta} rad, move_time: {self.move_time}s')

        # one-shot timer
        self.timer = self.create_timer(1.0, self._send_once)
        self.sent = False

    def _send_once(self):
        if self.sent:
            return
        self.sent = True

        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        # We send a relative wiggle on joint1 by using two points:
        # point1: all zeros at t=move_time
        # point2: joint1 back to zero at t=2*move_time
        # NOTE: If your controller expects absolute positions, this will move toward 0.
        # Safer approach is to read current joint_states first; we can add that next.

        p1 = JointTrajectoryPoint()
        p1.positions = [0.0] * len(self.joint_names)
        p1.positions[0] = self.delta
        p1.time_from_start.sec = int(self.move_time)
        p1.time_from_start.nanosec = int((self.move_time - int(self.move_time)) * 1e9)

        p2 = JointTrajectoryPoint()
        p2.positions = [0.0] * len(self.joint_names)
        p2.time_from_start.sec = int(2 * self.move_time)
        p2.time_from_start.nanosec = int(((2 * self.move_time) - int(2 * self.move_time)) * 1e9)

        msg.points = [p1, p2]

        self.pub.publish(msg)
        self.get_logger().info('Trajectory published (2 points). Exiting in 2s...')
        self.create_timer(2.0, lambda: rclpy.shutdown())


def main():
    rclpy.init()
    node = SendJointTraj()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
