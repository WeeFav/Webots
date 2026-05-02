from setuptools import setup

package_name = 'create_dataset'
data_files = []
data_files.append(('share/ament_index/resource_index/packages', ['resource/' + package_name]))
data_files.append(('share/' + package_name + '/launch', ['launch/robot_launch.py', 'launch/webots_launch.py']))
data_files.append(('share/' + package_name + '/worlds', ['worlds/city.wbt', 'worlds/city_net/sumo.edg.xml', 'worlds/city_net/sumo.net.xml', 'worlds/city_net/sumo.nod.xml', 'worlds/city_net/sumo.rou.alt.xml', 'worlds/city_net/sumo.rou.xml', 'worlds/city_net/sumo.sumocfg']))
data_files.append(('share/' + package_name + '/resource', ['resource/robot.urdf']))
data_files.append(('share/' + package_name, ['package.xml']))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user.name@mail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_robot_driver = create_dataset.robot_driver:main',
            'obstacle_avoider = create_dataset.obstacle_avoider:main',
            'lane_follower = webots_ros2_tesla.lane_follower:main',
        ],
    },
)