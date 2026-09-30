#!/bin/bash
# Run the robohand in Gazebo (gz sim 10), detached.
#   tools/launch_gazebo.sh [Right|Left] [gui|headless]
set +u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
HAND="${1:-Right}"
MODE="${2:-gui}"
LOG=/tmp/rh/gazebo_${HAND}_${MODE}.log
mkdir -p /tmp/rh

source /opt/ros/lyrical/setup.bash
cd "$ROOT/ros2_ws"
source install/setup.bash 2>/dev/null

URDF="$ROOT/ros2_ws/src/robohand_sim/urdf/robohand_${HAND}.urdf"
SDF=/tmp/rh/hand_${HAND}.sdf
gz sdf -p "$URDF" > "$SDF" 2>/dev/null
echo "converted URDF -> SDF ($(grep -c '<joint' "$SDF") joints)"

if [ "$MODE" = "headless" ]; then
  setsid gz sim -s -r "$SDF" > "$LOG" 2>&1 < /dev/null &
  echo "gz sim started SERVER-ONLY (no window) - log $LOG"
else
  setsid gz sim -r "$SDF" > "$LOG" 2>&1 < /dev/null &
  echo "gz sim started with GUI - log $LOG"
fi
