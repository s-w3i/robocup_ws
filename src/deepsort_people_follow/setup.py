from glob import glob
from setuptools import setup

package_name = 'deepsort_people_follow'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name, ['README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='usern',
    maintainer_email='usern@example.com',
    description='ROS2 DeepSORT-based people follow pipeline with 3D fusion and follow target output.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'deepsort_people_follow_node = deepsort_people_follow.deepsort_people_follow_node:main',
        ],
    },
)
