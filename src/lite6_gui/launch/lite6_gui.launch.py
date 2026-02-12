from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cmd_topic = LaunchConfiguration('cmd_topic')
    js_topic = LaunchConfiguration('js_topic')

    return LaunchDescription([
        DeclareLaunchArgument('cmd_topic', default_value='/joint_trajectory_controller/joint_trajectory'),
        DeclareLaunchArgument('js_topic', default_value='/joint_states'),
        Node(
            package='lite6_gui',
            executable='lite6_gui',
            name='lite6_gui',
            output='screen',
            parameters=[
                {'cmd_topic': cmd_topic},
                {'js_topic': js_topic},
            ],
        )
    ])
