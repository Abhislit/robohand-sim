#!/bin/bash
# Verify the robohand URDF simulates in ROS 2: launch it headless, then confirm
# that /joint_states is published, that the angles actually move, and that
# robot_state_publisher turns them into TF transforms. Works over SSH / in CI
# because RViz is disabled.
set +u
cd "$(dirname "$0")/../ros2_ws"
source /opt/ros/lyrical/setup.bash
source install/setup.bash

HAND="${1:-Right}"
LOG=/tmp/rh/launch_${HAND}.log

ros2 launch robohand_sim view_hand.launch.py \
    hand:="$HAND" rviz:=false animate:=true > "$LOG" 2>&1 &
LAUNCH_PID=$!
echo "launched (hand=$HAND), waiting for nodes..."

for _ in $(seq 1 45); do
  ros2 node list 2>/dev/null | grep -q robohand_bridge && break
  sleep 1
done

echo "=== nodes ==="
ros2 node list 2>/dev/null

echo
echo "=== joint angles: do they move? ==="
python3 - <<'PY'
import re, subprocess, sys

def sample():
    out = subprocess.run(
        ["ros2", "topic", "echo", "/joint_states", "--once", "--field", "position"],
        capture_output=True, text=True, timeout=30).stdout
    return [float(m) for m in re.findall(r"-?\d+\.\d+(?:e[-+]?\d+)?", out)]

reads = []
for _ in range(6):
    r = sample()
    if r and len(r) == 15:
        reads.append(r)
if not reads:
    print("  FAIL: could not read a 15-element JointState"); sys.exit(1)

print(f"  joints per message : {len(reads[0])}")
print(f"  first message      : {[round(v, 3) for v in reads[0]]}")
spread = [max(r[i] for r in reads) - min(r[i] for r in reads) for i in range(15)]
moving = [i for i, s in enumerate(spread) if s > 1e-3]
print(f"  samples read       : {len(reads)}")
print(f"  joints that moved  : {len(moving)}/15")
print(f"  max travel (rad)   : {max(spread):.4f}")
print(f"  all within [0,1.9] : {all(0.0 <= v <= 1.9 for v in reads[0])}")
sys.exit(0 if (len(reads) >= 3 and len(moving) >= 12 and max(spread) > 0.05) else 1)
PY
JOINT_OK=$?

echo
echo "=== TF: does robot_state_publisher build a kinematic tree? ==="
timeout 30 ros2 topic echo /tf --once 2>/dev/null | grep -c "frame_id" || true
TF_N=$(timeout 30 ros2 topic echo /tf --once 2>/dev/null | grep -c "child_frame_id" || true)
echo "  dynamic transforms in one /tf message: $TF_N"
[ "${TF_N:-0}" -gt 0 ] && TF_OK=0 || TF_OK=1

echo
echo "=== errors in launch log ==="
ERRS=$(grep -cE "\[ERROR\]" "$LOG")
echo "  $ERRS"

kill $LAUNCH_PID 2>/dev/null
sleep 2
pkill -f robot_state_pub 2>/dev/null
pkill -f robohand_bridg 2>/dev/null
sleep 1

echo
if [ "$JOINT_OK" = "0" ] && [ "$TF_OK" = "0" ] && [ "$ERRS" = "0" ]; then
  echo "RESULT: PASS - hand simulates in ROS 2 (joints move, TF resolves, no errors)"
else
  echo "RESULT: FAIL (joints=$JOINT_OK tf=$TF_OK errors=$ERRS)"
fi
