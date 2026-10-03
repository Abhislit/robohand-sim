#!/usr/bin/env bash
# Launch / inspect / stop the robohand Gazebo world.
#
#   tools/gazebo.sh start [gui|headless]   build the world and launch it
#   tools/gazebo.sh status                what is running
#   tools/gazebo.sh stop                  stop only robohand's Gazebo
#   tools/gazebo.sh selftest              verify the simulated fingers move
#
# Stops short of any Gazebo that is not ours: a world whose file path does not
# contain "robohand" or "rh/" is left alone, so a blackbox / inav replay
# running on the same machine keeps running.
set +u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="$ROOT/.venv/bin/python"
ROS_SETUP=/opt/ros/lyrical/setup.bash
WORLD_DIR="${ROBOHAND_GZ_DIR:-$ROOT/.gazebo}"
WORLD_SDF="$WORLD_DIR/robohand_world.sdf"
LOG="$WORLD_DIR/gazebo.log"
HAND=Right

ours() {   # true if this gz-sim-main is serving one of our worlds
  local cmd; cmd=$(tr '\0' ' ' < /proc/"$1"/cmdline 2>/dev/null)
  case "$cmd" in *robohand*|*/rh/*) return 0 ;; *) return 1 ;; esac
}

build_world() {
  mkdir -p "$WORLD_DIR"
  # Generate to a temp file, re-parse it, and only then move it into place, so
  # a generator that raises can never leave a 0-byte world behind for gz sim to
  # choke on.
  local tmp="$WORLD_SDF.tmp"
  if ! "$PY" - "$HAND" "$tmp" <<'PY'
import sys
import xml.etree.ElementTree as ET
from robohand.gazebo_world import build_world

hand, out = sys.argv[1], sys.argv[2]
xml = build_world(hand, with_arm=True)
ET.fromstring(xml)                     # fail loudly on a malformed SDF
with open(out, "w") as fh:
    fh.write(xml)
print(f"wrote {out} ({len(xml)} bytes)")
PY
  then
    echo "ERROR: world generation failed; leaving $WORLD_SDF untouched" >&2
    rm -f "$tmp"
    return 1
  fi
  mv "$tmp" "$WORLD_SDF"
}

start() {
  local mode="${1:-gui}"
  # Rebuild when missing or empty: a 0-byte world file makes gz sim fail with
  # a bare "Error parsing XML ... EMPTY_DOCUMENT" and no useful context.
  [ -s "$WORLD_SDF" ] || build_world || return 1
  # shellcheck disable=SC1090
  source "$ROS_SETUP" >/dev/null 2>&1
  if [ "$mode" = "headless" ]; then
    setsid gz sim -s -r "$WORLD_SDF" > "$LOG" 2>&1 < /dev/null &
  else
    setsid gz sim -r "$WORLD_SDF" > "$LOG" 2>&1 < /dev/null &
  fi
  echo "gz sim starting ($mode) - world: $WORLD_SDF"
  echo "log: $LOG"
  echo "wait ~30s, then:  tools/gazebo.sh status"
}

stop() {
  local n=0 p
  for p in $(pgrep -f "gz-sim-main" 2>/dev/null) $(pgrep -f "gz-sim-gui-client" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    if ours "$p"; then kill "$p" 2>/dev/null && n=$((n+1)); fi
  done
  sleep 2
  for p in $(pgrep -f "gazebo_pub.py" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    kill "$p" 2>/dev/null && n=$((n+1))
  done
  echo "stopped $n robohand gazebo process(es); other gazebo instances untouched"
}

status() {
  # shellcheck disable=SC1090
  source "$ROS_SETUP" >/dev/null 2>&1
  local found=0 p
  echo "=== robohand gazebo ==="
  for p in $(pgrep -f "gz-sim-main" 2>/dev/null); do
    if ours "$p"; then echo "  running: $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)"; found=1; fi
  done
  [ "$found" = 0 ] && echo "  not running"
  if [ "$found" = 1 ]; then
    echo "  models: $(timeout 20 gz model --list 2>/dev/null | tail -n +3 | tr -d ' -' | tr '\n' ' ')"
    echo "  errors in log: $(grep -a -icE '\[error\]' "$LOG" 2>/dev/null || echo 0)"
    echo "  joint cmd topics: $(timeout 20 gz topic --list 2>/dev/null | grep -c cmd_pos)"
  fi
  echo "=== other gazebo (left alone) ==="
  for p in $(pgrep -f "gz-sim-main" 2>/dev/null); do
    ours "$p" || echo "  $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-90)"
  done
}

case "${1:-start}" in
  start) start "${2:-gui}" ;;
  stop)  stop ;;
  status) status ;;
  selftest) "$PY" "$HERE/verify_gazebo.py" ;;
  build) build_world ;;
  *) echo "usage: $0 {start [gui|headless]|stop|status|selftest|build}"; exit 2 ;;
esac
