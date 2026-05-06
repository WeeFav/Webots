"""webots_ros2 package setup file."""

from setuptools import setup


package_name = 'test_tesla'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/launch', ['launch/robot_launch.py', 'launch/webots_launch.py']))
data_files.append(('share/' + package_name + '/worlds', [
    'worlds/tesla_world.wbt',
]))
data_files.append(('share/' + package_name + '/resource', [
    'resource/tesla_webots.urdf'
]))
data_files.append(('share/' + package_name, ['package.xml']))


setup(
    name=package_name,
    version='2025.0.1',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools', 'launch'],
    zip_safe=True,
    author='Cyberbotics',
    author_email='support@cyberbotics.com',
    maintainer='Cyberbotics',
    maintainer_email='support@cyberbotics.com',
    keywords=['ROS', 'Webots', 'Robot', 'Simulation', 'Examples'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Tesla ROS2 interface for Webots.',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_follower = test_tesla.lane_follower:main'
        ],
        'launch.frontend.launch_extension': ['launch_ros = launch_ros']
    }
)