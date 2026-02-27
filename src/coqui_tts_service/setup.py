from setuptools import setup

package_name = 'coqui_tts_service'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='usern',
    maintainer_email='usern@local',
    description='ROS2 Coqui TTS service node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'coqui_tts_service_node = coqui_tts_service.coqui_tts_service_node:main',
            'coqui_talking_face_action_node = coqui_tts_service.coqui_talking_face_action_node:main',
        ],
    },
)
