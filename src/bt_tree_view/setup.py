from glob import glob

from setuptools import setup

package_name = "bt_tree_view"

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
    description="ROS 2 monitor node for live behaviour tree snapshots.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bt_monitor_node = bt_tree_view.bt_monitor_node:main",
            "bt_monitor_ui = bt_tree_view.bt_monitor_ui:main",
        ],
    },
)
