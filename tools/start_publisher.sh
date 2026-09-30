#!/bin/bash
# Start ONLY robot_state_publisher for the robohand, so the live bridge is the
# single source of /joint_states (no slider GUI competing with it).
set +u
cd "$(dirname "$0")/../ros2_ws"
source /opt/ros/lyrical/setup.bash
source install/setup.bash
HAND="${1:-Right}"
URDF="$PWD/src/robohand_sim/urdf/robohand_${HAND}.urdf"

pkill -f "joint_state_publisher_gui" 2>/dev/null
pkill -f "robot_state_pub" 2>/dev/null
sleep 2

setsid ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$(cat "$URDF")" \
    > /tmp/rh/rsp.log 2>&1 < /dev/null &
echo "robot_state_publisher started (hand=$HAND)"
