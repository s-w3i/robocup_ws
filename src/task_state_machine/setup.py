from glob import glob

from setuptools import setup

package_name = "task_state_machine"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="usern",
    maintainer_email="usern@example.com",
    description="YASMIN ROS 2 task state machines.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "carry_my_luggage_state_machine_node = task_state_machine.carry_my_luggage_state_machine_node:main",
            "mock_location_image_service_node = task_state_machine.mock_location_image_service_node:main",
            "receptionist_state_machine_node = task_state_machine.receptionist_state_machine_node:main",
            "telegram_lost_found_node = task_state_machine.telegram_lost_found_node:main",
            "telegram_food_drink_state_machine_node = task_state_machine.telegram_food_drink_state_machine_node:main",
        ],
    },
)
