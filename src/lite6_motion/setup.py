from setuptools import find_packages, setup

package_name = 'lite6_motion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='r91',
    maintainer_email='kunlun1988@me.com',
    description='Safety-gated ROS2 motion API for Lite6 robot arm',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'motion_server = lite6_motion.motion_server:main',
        ],
    },
)
