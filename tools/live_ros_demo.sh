#!/bin/bash
# Full live loop: webcam hand tracking -> ROS 2 /joint_states -> RViz + Gazebo.
#
#   tools/live_ros_demo.sh [Right|Left]
#
# Starts robot_state_publisher, then runs the webcam app with --ros so your
# actual hand drives the simulated hand. Light the room or it will not track.
set +u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
HAND="${1:-Right}"
PY="$ROOT/.venv/bin/python"

cd "$ROOT/ros2_ws"
source /opt/ros/lyrical/setup.bash
source install/setup.bash

URDF="$ROOT/ros2_ws/src/robohand_sim/urdf/robohand_${HAND}.urdf"

# robot_state_publisher: URDF -> TF. This is the simulator's kinematics.
setsid ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$(cat "$URDF")" \
    > /tmp/rh/live_rsp.log 2>&1 < /dev/null &
echo "[1/3] robot_state_publisher up (hand=$HAND)"

# RViz to look at it.
setsid ros2 run rviz2 rviz2 -d \
    "$ROOT/ros2_ws/src/robohand_sim/rviz/hand.rviz" \
    > /tmp/rh/live_rviz.log 2>&1 < /dev/null &
echo "[2/3] RViz starting..."

sleep 6

# The app: webcam -> flexion values -> stdin -> bridge -> /joint_states.
echo "[3/3] starting webcam app with live ROS streaming"
echo
cd "$ROOT"
setsid "$PY" -u -m robohand --source 0 --ros "$URDF" --ros-hand "$HAND" \
    > /tmp/rh/live_app.log 2>&1 < /dev/null &
echo
echo "logs:  /tmp/rh/live_app.log   /tmp/rh/live_rviz.log   /tmp/rh/live_rsp.log"
echo "stop:  /tmp/rh/stop_ros.sh"
