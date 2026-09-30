#!/bin/bash
# Launch the robohand simulation in RViz, detached, and report what came up.
#   tools/launch_ros2_sim.sh [Right|Left] [animate|sliders]
set +u
cd "$(dirname "$0")/../ros2_ws"
source /opt/ros/lyrical/setup.bash
source install/setup.bash

HAND="${1:-Right}"
MODE="${2:-animate}"
LOG=/tmp/rh/sim_${HAND}_${MODE}.log
mkdir -p /tmp/rh

if [ "$MODE" = "sliders" ]; then
  ARGS=(hand:="$HAND" animate:=false)
else
  ARGS=(hand:="$HAND" animate:=true)
fi

echo "launching: hand=$HAND mode=$MODE"
setsid ros2 launch robohand_sim view_hand.launch.py "${ARGS[@]}" \
    > "$LOG" 2>&1 < /dev/null &
echo "log: $LOG"
