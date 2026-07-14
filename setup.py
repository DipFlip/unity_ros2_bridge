from setuptools import find_packages, setup


package_name = "unity_ros2_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Emil Rofors",
    maintainer_email="erofors@lbl.gov",
    description="TCP bridges between Unity simulations and ROS 2.",
    license="TBD",
    entry_points={
        "console_scripts": [
            "unity_camera_bridge = unity_ros2_bridge.unity_camera_bridge:main",
            "unity_spot_bridge = unity_ros2_bridge.unity_camera_bridge:main",
        ],
    },
)
