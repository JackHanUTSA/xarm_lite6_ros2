from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument('mode', default_value='sim'),
            DeclareLaunchArgument('manifest_path', default_value=''),
            Node(
                package='epuck2_sim_real_control',
                executable='project_info',
                name='epuck2_project_info',
                output='screen',
                parameters=[
                    {
                        'mode': LaunchConfiguration('mode'),
                        'manifest_path': LaunchConfiguration('manifest_path'),
                    }
                ],
            ),
        ]
    )
