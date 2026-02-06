import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


class SendJointTraj(Node):
    def __init__(self):
        super().__init__('send_joint_traj')

        self.declare_parameter('topic', '/xarm6_traj_controller/joint_trajectory')
        self.declare_parameter('joint_names', ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'])
        self.declare_parameter('delta_joint1', 0.10)  # radians
        self.declare_parameter('move_time', 8.0)      # seconds
        self.declare_parameter('return_back', True)

        self.topic = self.get_parameter('topic').value
        self.joint_names = list(self.get_parameter('joint_names').value)
        self.delta = float(self.get_parameter('delta_joint1').value)
        self.move_time = float(self.get_parameter('move_time').value)
        self.return_back = bool(self.get_parameter('return_back').value)

        self.pub = self.create_publisher(JointTrajectory, self.topic, 10)
        self.sub = self.create_subscription(JointState, '/joint_states', self._on_js, 10)

        self._last_js = None
        self._sent = False

        self.get_logger().info(f'Publishing to {self.topic}')
        self.get_logger().info('Waiting for /joint_states...')

    def _on_js(self, msg: JointState):
        self._last_js = msg
        if self._sent:
            return

        # Need all desired joints present
        name_to_pos = {n: p for n, p in zip(msg.name, msg.position)}
        if not all(n in name_to_pos for n in self.joint_names):
            missing = [n for n in self.joint_names if n not in name_to_pos]
            self.get_logger().warn(f'Missing joints in /joint_states: {missing}')
            return

        current = [float(name_to_pos[n]) for n in self.joint_names]
        target = current.copy()
        target[0] = target[0] + self.delta

        traj = JointTrajectory()
        traj.joint_names = self.joint_names

        p1 = JointTrajectoryPoint()
        p1.positions = target
        p1.time_from_start.sec = int(self.move_time)
        p1.time_from_start.nanosec = int((self.move_time - int(self.move_time)) * 1e9)

        traj.points = [p1]

        if self.return_back:
            p2 = JointTrajectoryPoint()
            p2.positions = current
            t2 = 2.0 * self.move_time
            p2.time_from_start.sec = int(t2)
            p2.time_from_start.nanosec = int((t2 - int(t2)) * 1e9)
            traj.points.append(p2)

        self.pub.publish(traj)
        self._sent = True
        self.get_logger().info(f'Sent relative trajectory: joint1 +{self.delta} rad (and back={self.return_back}).')
        self.get_logger().info('Exiting in 2s...')
        self.create_timer(2.0, lambda: rclpy.shutdown())


def main():
    rclpy.init()
    node = SendJointTraj()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
