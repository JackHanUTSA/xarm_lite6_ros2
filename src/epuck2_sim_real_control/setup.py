from setuptools import find_packages, setup

package_name = 'epuck2_sim_real_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', ['config/default_session.yaml']),
        ('share/' + package_name + '/launch', ['launch/epuck2_sim_real_control.launch.py']),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='r91',
    maintainer_email='kunlun1988@me.com',
    description='Scaffold for e-puck2 sim-to-real control workflows in ROS 2',
    license='TODO: License declaration',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'project_info = epuck2_sim_real_control.project_info:main',
        ],
    },
)
