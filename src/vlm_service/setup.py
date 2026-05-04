from glob import glob

from setuptools import setup

package_name = 'vlm_service'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'requests', 'py_trees'],
    zip_safe=True,
    maintainer='usern',
    maintainer_email='usern@example.com',
    description='ROS2 VLM service node.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vlm_query_service_node = vlm_service.vlm_query_service_node:main',
            'camera_snapshot_service_node = vlm_service.camera_snapshot_service_node:main',
            'lost_found_vlm_check_node = vlm_service.lost_found_vlm_check_node:main',
            'food_drink_sort_service_node = vlm_service.food_drink_sort_service_node:main',
            'ask_name_and_drink_action_node = vlm_service.ask_name_and_drink_action_node:main',
            'describe_human_from_camera0_node = vlm_service.describe_human_from_camera0_node:main',
            'describe_human_action_node = vlm_service.describe_human_action_node:main',
        ],
    },
)
