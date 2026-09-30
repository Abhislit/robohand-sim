"""Run the real app loop with the GUI stubbed out and save frames to disk.

Lets the full pipeline (tracking -> smoothing -> FK -> render -> HUD) be
verified on machines with no usable display, and produces reference images.

    python tools/selftest.py --source demo --frames 90
    python tools/selftest.py --source 0 --frames 30
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robohand.app import Config, RoboHandApp  # noqa: E402

SAVED: list[np.ndarray] = []


def _stub_gui(frame_budget: int, save_every: int, out_dir: str):
    """Replace the OpenCV GUI calls so the loop can run headless."""
    state = {"n": 0}

    def wait_key(_delay):
        state["n"] += 1
        if state["n"] % save_every == 0:
            path = os.path.join(out_dir, f"frame_{state['n']:04d}.png")
            cv2.imwrite(path, SAVED[-1])
            print(f"  saved {path}")
        if state["n"] >= frame_budget:
            return ord("q")
        return 255

    def imshow(_name, _img):
        pass

    def named_window(*_a, **_k):
        pass

    def destroy(*_a, **_k):
        pass

    cv2.waitKey = wait_key
    cv2.imshow = imshow
    cv2.namedWindow = named_window
    cv2.resizeWindow = lambda *a, **k: None
    cv2.destroyAllWindows = destroy
    cv2.destroyWindow = destroy


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="demo")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--save-every", type=int, default=30)
    p.add_argument("--out", default="")
    args = p.parse_args()

    demo = args.source.lower() == "demo"
    out_dir = args.out or os.path.join("tools", "out", "selftest")
    os.makedirs(out_dir, exist_ok=True)

    cfg = Config(
        source=0 if demo else (int(args.source) if args.source.isdigit() else args.source),
        demo=demo,
    )
    _stub_gui(args.frames, args.save_every, out_dir)

    application = RoboHandApp(cfg)
    real_compose = application._compose

    def spy(frame, hands, smoothed, dt):
        img = real_compose(frame, hands, smoothed, dt)
        SAVED.append(img)
        return img

    application._compose = spy
    rc = application.run()
    print(f"selftest finished rc={rc}, {len(SAVED)} frames composed")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
