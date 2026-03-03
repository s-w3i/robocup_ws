from glob import glob
from setuptools import setup

package_name = 'yoloe_detection_service'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='usern',
    maintainer_email='usern@example.com',
    description='ROS2 YOLOE text-prompt detection service.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yoloe_detection_service_node = yoloe_detection_service.yoloe_detection_service_node:main',
            'yoloe_detection_client = yoloe_detection_service.yoloe_detection_client:main',
            'yoloe_tracking_control_client = yoloe_detection_service.yoloe_tracking_control_client:main',
            'yoloe_pointed_detection_service_node = yoloe_detection_service.yoloe_pointed_detection_service_node:main',
            'yoloe_pointed_detection_client = yoloe_detection_service.yoloe_pointed_detection_client:main',
            'yoloe_vlm_pointed_detection_service_node = yoloe_detection_service.yoloe_vlm_pointed_detection_service_node:main',
        ],
    },
)
