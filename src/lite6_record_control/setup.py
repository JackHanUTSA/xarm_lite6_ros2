from setuptools import find_packages, setup

package_name = 'lite6_record_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='r91',
    maintainer_email='kunlun1988@me.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'triple_panel_record = lite6_record_control.triple_panel_record:main',
        'ros_image_snapshot = lite6_record_control.ros_image_snapshot:main',
            'run_and_record = lite6_record_control.run_and_record:main',
        ],
    },
)
