from setuptools import find_packages, setup

package_name = 'lite6_imitation_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/calibration_schema.yaml']),
    ],
    install_requires=['setuptools', 'numpy', 'PyYAML'],
    zip_safe=True,
    maintainer='r91',
    maintainer_email='kunlun1988@me.com',
    description='FreeMoCap human arm feature export bridge for Lite6 imitation workflows',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'export_human_arm_demo = lite6_imitation_bridge.export_human_arm_demo:main',
            'calibrate_human_to_robot_frame = lite6_imitation_bridge.calibrate_human_to_robot_frame:main',
            'export_lite6_targets = lite6_imitation_bridge.export_lite6_targets:main',
            'preview_targets = lite6_imitation_bridge.preview_targets:main',
            'publish_joint_command_preview = lite6_imitation_bridge.publish_joint_command_preview:main',
            'execute_lite6_targets = lite6_imitation_bridge.execute_lite6_targets:main',
        ],
    },
)
