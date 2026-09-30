"""View the robohand in RViz.

    ros2 launch robohand_sim view_hand.launch.py                 # sliders to pose it
    ros2 launch robohand_sim view_hand.launch.py animate:=true   # auto open/close
    ros2 launch robohand_sim view_hand.launch.py hand:=Left
    ros2 launch robohand_sim view_hand.launch.py gazebo:=true    # also Gazebo
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    FileContent,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node

# A LaunchConfiguration is a substitution, not a string, so an f-string would
# embed its repr. PythonExpression defers the join to launch time; the quotes
# around the substituted value are what make the generated expression valid
# Python ('robohand_' + 'Right' + '.urdf').
_URDF_NAME = ["'robohand_' + '", LaunchConfiguration("hand"), "' + '.urdf'"]


def generate_launch_description():
    pkg = get_package_share_directory("robohand_sim")
    animate = LaunchConfiguration("animate")

    urdf = PathJoinSubstitution(
        [pkg, "urdf", PythonExpression(_URDF_NAME)]
    )
    rviz_cfg = os.path.join(pkg, "rviz", "hand.rviz")

    declared = [
        DeclareLaunchArgument("hand", default_value="Right",
                              choices=["Right", "Left"]),
        DeclareLaunchArgument("animate", default_value="false",
                              description="scripted open/close instead of sliders"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("gazebo", default_value="false",
                              description="also spawn the hand in Gazebo"),
    ]

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": FileContent(urdf)}],
        ),
        # Sliders only make sense when nothing else is publishing joint states.
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            name="joint_state_publisher_gui",
            output="screen",
            condition=UnlessCondition(animate),
        ),
        Node(
            package="robohand_sim",
            executable="robohand_bridge.py",
            name="robohand_bridge",
            output="screen",
            arguments=[
                "--urdf", urdf,
                "--hand", LaunchConfiguration("hand"),
                "--source", "demo",
            ],
            condition=IfCondition(animate),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_cfg],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
        Node(
            package="ros_gz_sim",
            executable="gz_sim",
            name="gz_sim",
            output="screen",
            arguments=["-r", "-v", "3", "-s", "libgz-sim.so"],
            condition=IfCondition(LaunchConfiguration("gazebo")),
        ),
        Node(
            package="ros_gz_sim",
            executable="create",
            name="spawn_hand",
            output="screen",
            arguments=[urdf, "-name", "robohand", "-topic", "/robot_description"],
            condition=IfCondition(LaunchConfiguration("gazebo")),
        ),
    ]

    return LaunchDescription(declared + nodes)
