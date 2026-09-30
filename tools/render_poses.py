"""Render a pose sheet so the rig geometry can be eyeballed without a camera."""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robohand.robot_hand import RobotHand  # noqa: E402
from robohand.renderer import Renderer  # noqa: E402

POSES = {
    "open": {f: 0.05 for f in ("thumb", "index", "middle", "ring", "pinky")},
    "half": {"thumb": 0.35, "index": 0.5, "middle": 0.5, "ring": 0.5, "pinky": 0.5},
    "fist": {f: 1.0 for f in ("thumb", "index", "middle", "ring", "pinky")},
    "point": {"thumb": 0.9, "index": 0.05, "middle": 1.0, "ring": 1.0, "pinky": 1.0},
    "peace": {"thumb": 0.9, "index": 0.05, "middle": 0.05, "ring": 1.0, "pinky": 1.0},
    "pinch": {"thumb": 0.7, "index": 0.6, "middle": 0.35, "ring": 0.3, "pinky": 0.3},
}


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    os.makedirs(out_dir, exist_ok=True)

    w, h = 420, 460
    r = Renderer(w, h, supersample=2)
    tiles = []
    for name, curls in POSES.items():
        hand = RobotHand("Right")
        signal = {f"{f}_flex": v for f, v in curls.items()}
        signal["thumb_opposition"] = 1.0 if curls["index"] > 0.4 else 0.0
        hand.set_pose(signal)
        img = r.render(hand.solve())
        cv2.putText(img, name.upper(), (14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (235, 235, 235), 2, cv2.LINE_AA)
        tiles.append(img)

    top = np.hstack(tiles[:3])
    bottom = np.hstack(tiles[3:])
    sheet = np.vstack([top, bottom])
    path = os.path.join(out_dir, "pose_sheet.png")
    cv2.imwrite(path, sheet)
    print(f"wrote {path}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
