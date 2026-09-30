"""A small dependency-free 3D renderer.

Projects the robot hand with a pinhole camera and rasterises it with OpenCV
using a painter's algorithm (back-to-front, sorted by depth). Cylinders are
shaded by drawing three stacked strokes -- outline, body, specular band -- which
reads as a rounded metal surface and needs no z-buffer or OpenGL context, so it
runs anywhere (including headless / over SSH).
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .math3d import cross, dot, norm, vec
from .robot_hand import Primitive

LIGHT_DIR = norm(vec(-0.45, 0.78, 0.62))
AMBIENT = 0.44
DIFFUSE = 0.66
_SHADOW_RGB = np.array([8.0, 8.0, 11.0], np.float32)


class Camera3D:
    """Pinhole camera orbiting a target point."""

    def __init__(
        self,
        width: int,
        height: int,
        fov_deg: float = 38.0,
        distance: float = 0.50,
        target: tuple = (0.0, -0.022, 0.0),
        yaw: float = -0.30,
        pitch: float = 0.12,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fov_deg = float(fov_deg)
        self.distance = float(distance)
        self.target = np.asarray(target, dtype=float)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self._update()

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)
        self._update()

    def orbit(self, yaw: float, pitch: float) -> None:
        self.yaw, self.pitch = float(yaw), float(pitch)
        self._update()

    def _update(self) -> None:
        self.focal = (self.height * 0.5) / math.tan(math.radians(self.fov_deg) * 0.5)
        self.cx, self.cy = self.width * 0.5, self.height * 0.5
        cp = math.cos(self.pitch)
        self.position = self.target + self.distance * vec(
            math.sin(self.yaw) * cp, math.sin(self.pitch), math.cos(self.yaw) * cp
        )
        self.forward = norm(self.target - self.position)
        self.right = norm(cross(self.forward, vec(0.0, 1.0, 0.0)))
        self.up = cross(self.right, self.forward)

    def view_coords(self, pts: np.ndarray) -> np.ndarray:
        rel = np.asarray(pts, dtype=float).reshape(-1, 3) - self.position
        return np.column_stack(
            [rel @ self.right, rel @ self.up, rel @ self.forward]
        )

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points -> (pixel xy, depth). Depth <= 0 means behind camera."""
        v = self.view_coords(pts)
        z = np.maximum(v[:, 2], 1e-4)
        uv = np.column_stack(
            [self.cx + self.focal * v[:, 0] / z, self.cy - self.focal * v[:, 1] / z]
        )
        return uv, v[:, 2]

    def project_one(self, p) -> tuple[float, float, float]:
        uv, d = self.project(np.asarray(p, dtype=float).reshape(1, 3))
        return float(uv[0, 0]), float(uv[0, 1]), float(d[0])

    def radius_px(self, r: float, depth: float) -> float:
        return self.focal * float(r) / max(float(depth), 1e-4)

    def to_camera(self, p) -> np.ndarray:
        return np.asarray(p, dtype=float) - self.position


def _shade(color: tuple, k: float) -> tuple:
    """Scale an RGB triple. Pure-python on purpose - this is called ~700x/frame."""
    k = 0.0 if k < 0.0 else (1.35 if k > 1.35 else k)
    b, g, r = color
    return (
        min(255, int(b * k + 0.5)),
        min(255, int(g * k + 0.5)),
        min(255, int(r * k + 0.5)),
    )


class Renderer:
    """Rasterises `Primitive` lists into a BGR image."""

    def __init__(self, width: int, height: int, supersample: int = 2) -> None:
        self.width = int(width)
        self.height = int(height)
        self.ss = max(1, int(supersample))
        self.camera = Camera3D(self.width * self.ss, self.height * self.ss)
        self.canvas = np.zeros((self.height * self.ss, self.width * self.ss, 3), np.uint8)
        self._bg: np.ndarray | None = None
        self._shadow_tpl: np.ndarray | None = None

    # -- public -------------------------------------------------------------
    def resize(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)
        self.camera.resize(self.width * self.ss, self.height * self.ss)
        self.canvas = np.zeros((self.height * self.ss, self.width * self.ss, 3), np.uint8)
        self._bg = None

    def orbit(self, yaw: float, pitch: float) -> None:
        self.camera.orbit(yaw, pitch)

    def render(
        self,
        primitives: list[Primitive],
        grid: bool = True,
        shadow: bool = True,
        floor_y: float = -0.135,
    ) -> np.ndarray:
        self._background()
        if grid:
            self._floor(floor_y)
        if primitives:
            if shadow:
                self._shadow(primitives, floor_y)
            self._draw_sorted(primitives)
        return self._downsample()

    # -- stages -------------------------------------------------------------
    def _background(self) -> None:
        """Static backdrop. Blurring a canvas-sized glow costs ~1 s per frame,
        so it is rendered once and blitted thereafter."""
        if self._bg is not None:
            self.canvas[:] = self._bg
            return
        h, w = self.canvas.shape[:2]
        top = np.array([46, 40, 34], np.float32)
        bottom = np.array([20, 22, 27], np.float32)
        t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        grad = (top[None, :] * (1.0 - t) + bottom[None, :] * t).astype(np.uint8)
        bg = np.repeat(grad[:, None, :], w, axis=1)

        # Build the glow small and upscale: visually identical, far cheaper.
        gs = 8
        small = np.zeros((max(2, h // gs), max(2, w // gs)), np.float32)
        cv2.circle(small, (w // (2 * gs), int(h * 0.42 / gs)),
                   max(2, int(min(w, h) * 0.34 / gs)), 1.0, -1, cv2.LINE_AA)
        glow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        glow = cv2.GaussianBlur(glow, (0, 0), min(w, h) * 0.10)[..., None]
        self._bg = np.clip(
            bg.astype(np.float32) + glow * np.array([26.0, 24.0, 20.0]), 0, 255
        ).astype(np.uint8)
        self.canvas[:] = self._bg

    def _floor(self, floor_y: float) -> None:
        cam = self.camera
        extent, step = 0.42, 0.06
        for i in range(int(-extent / step), int(extent / step) + 1):
            t = i * step
            for a, b in (
                (vec(t, floor_y, -extent), vec(t, floor_y, extent)),
                (vec(-extent, floor_y, t), vec(extent, floor_y, t)),
            ):
                uv, d = cam.project(np.array([a, b]))
                if np.any(d <= 0.02):
                    continue
                fade = float(np.clip(1.0 - d.mean() / (cam.distance * 2.4), 0.06, 1.0))
                col = _shade((58, 52, 44), fade * 1.5)
                cv2.line(
                    self.canvas, tuple(uv[0].astype(int)), tuple(uv[1].astype(int)),
                    col, max(1, self.ss), cv2.LINE_AA,
                )

    def _shadow(self, prims: list[Primitive], floor_y: float) -> None:
        """Soft contact shadow from a cached blurred template that is merely
        rescaled and alpha-composited."""
        cam = self.camera
        pts = []
        for p in prims:
            for v in (p.p0, p.p1, p.center):
                if v is not None:
                    pts.append(v)
        if not pts:
            return
        arr = np.asarray(pts, dtype=float)
        cx, cy, cz = arr.mean(axis=0)

        lift = max(0.0, float(cy) - floor_y)
        world_r = 0.045 + 0.55 * min(lift, 0.25)          # grows as the hand rises
        ux, uy, d = cam.project_one(vec(cx, floor_y + 0.001, cz))
        if d <= 0.02:
            return
        r = min(cam.radius_px(world_r, d), self.height)
        rw, rh = int(r * 3.2), int(r * 0.9)
        if rw < 4 or rh < 2:
            return

        if self._shadow_tpl is None:
            t = np.zeros((64, 192), np.float32)
            cv2.ellipse(t, (96, 32), (88, 24), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            self._shadow_tpl = cv2.GaussianBlur(t, (0, 0), 12.0)

        patch = cv2.resize(self._shadow_tpl, (rw, rh), interpolation=cv2.INTER_LINEAR)
        x0, y0 = int(ux - rw / 2.0), int(uy - rh / 2.0)
        h, w = self.canvas.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x0 + rw), min(h, y0 + rh)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        a = (patch[: y1 - y0, : x1 - x0] * 0.55)[..., None]
        roi = self.canvas[y0:y1, x0:x1].astype(np.float32)
        self.canvas[y0:y1, x0:x1] = (roi * (1.0 - a) + _SHADOW_RGB * a).astype(np.uint8)

    def _draw_sorted(self, prims: list[Primitive]) -> None:
        cam = self.camera
        items = []
        for p in prims:
            pts = (
                np.array([p.p0, p.p1])
                if p.kind == "capsule"
                else _box_corners(p.center, p.half, p.rot)
            )
            if pts is None:
                continue
            _, d = cam.project(pts)
            if np.all(d <= 0.01):
                continue
            items.append((float(np.mean(d)) - p.bias, p))
        items.sort(key=lambda it: -it[0])          # far -> near
        for _, p in items:
            if p.kind == "capsule":
                self._capsule(p)
            else:
                self._box(p)

    # -- primitives ---------------------------------------------------------
    def _capsule(self, p: Primitive) -> None:
        cam = self.camera
        uvp, d = cam.project(np.array([p.p0, p.p1], dtype=float))
        (u0, v0), (u1, v1) = uvp
        dmid = max(float(d.mean()), 1e-3)
        rmid = max(cam.radius_px((p.r0 + p.r1) * 0.5, dmid), 0.7)

        axis = norm(np.asarray(p.p1, float) - np.asarray(p.p0, float))
        mid = (np.asarray(p.p0, float) + np.asarray(p.p1, float)) * 0.5
        radius = max(p.r0, p.r1, 1e-4)

        to_cam = norm(cam.to_camera(mid) - axis * dot(cam.to_camera(mid), axis))
        n_lit = LIGHT_DIR - axis * dot(LIGHT_DIR, axis)
        if float(np.linalg.norm(n_lit)) < 1e-6:
            n_lit = to_cam.copy()
        n_lit = norm(n_lit)
        if dot(n_lit, to_cam) < 0:
            n_lit = -n_lit

        diff = max(0.0, dot(n_lit, LIGHT_DIR))
        rim = max(0.0, dot(to_cam, LIGHT_DIR)) * 0.25
        body = _shade(p.color, AMBIENT + DIFFUSE * diff * 0.85 + rim)
        outline = _shade(p.color, AMBIENT * 0.42)
        spec = _shade(p.color, AMBIENT + DIFFUSE * 1.02 + 0.22)

        a, b = (int(round(u0)), int(round(v0))), (int(round(u1)), int(round(v1)))
        w_out = int(round(2.0 * rmid + 2.0 * self.ss * max(p.edge, 0.4)))
        cv2.line(self.canvas, a, b, outline, w_out, cv2.LINE_AA)
        cv2.line(self.canvas, a, b, body, int(round(2.0 * rmid)), cv2.LINE_AA)
        for pt, r in ((a, p.r0), (b, p.r1)):
            rr = max(cam.radius_px(r, dmid), 0.7)
            cv2.circle(self.canvas, pt, int(round(rr + self.ss)), outline, -1, cv2.LINE_AA)
            cv2.circle(self.canvas, pt, int(round(rr)), body, -1, cv2.LINE_AA)

        if p.metal > 0.05 and rmid > 3.0 * self.ss:
            hp, _ = cam.project(mid + n_lit * radius * 0.52)
            centre = uvp.mean(axis=0)
            off = (float(hp[0, 0] - centre[0]), float(hp[0, 1] - centre[1]))
            a2 = (int(round(u0 + off[0])), int(round(v0 + off[1])))
            b2 = (int(round(u1 + off[0])), int(round(v1 + off[1])))
            t = int(round(2.0 * rmid * 0.30 * p.metal))
            if t >= 1:
                cv2.line(self.canvas, a2, b2, spec, t, cv2.LINE_AA)
                for pt in (a2, b2):
                    cv2.circle(self.canvas, pt, max(t // 2, 1), spec, -1, cv2.LINE_AA)

    def _box(self, p: Primitive) -> None:
        cam = self.camera
        corners = _box_corners(p.center, p.half, p.rot)
        if corners is None:
            return
        uv, d = cam.project(corners)
        if np.any(d <= 0.01):
            return

        # Cyclic vertex rings (a bow-tie order would make fillConvexPoly tear).
        faces = [
            (0, 1, 2, 3),   # -Z
            (4, 5, 6, 7),   # +Z
            (0, 1, 5, 4),   # -Y
            (3, 2, 6, 7),   # +Y
            (0, 4, 7, 3),   # -X
            (1, 5, 6, 2),   # +X
        ]
        drawn = []
        box_c = np.asarray(p.center, float)
        for idx in faces:
            quad = corners[list(idx)]
            fc = quad.mean(axis=0)
            n = _face_normal(quad)
            if dot(n, fc - box_c) < 0.0:              # force outward winding
                n = -n
            if dot(n, cam.to_camera(fc)) <= 0.0:
                continue                              # back-face culling
            diff = max(0.0, dot(n, LIGHT_DIR))
            drawn.append((float(cam.view_coords(fc.reshape(1, 3))[0, 2]) - p.bias,
                          (_shade(p.color, AMBIENT + DIFFUSE * diff * 0.9), idx)))
        drawn.sort(key=lambda it: -it[0])
        for _, (col, idx) in drawn:
            pts = np.array([uv[i] for i in idx], np.int32).reshape(-1, 1, 2)
            cv2.fillConvexPoly(self.canvas, pts, col, cv2.LINE_AA)
            cv2.polylines(self.canvas, [pts], True, _shade(p.color, AMBIENT * 0.5),
                          max(1, self.ss), cv2.LINE_AA)

    def _downsample(self) -> np.ndarray:
        if self.ss == 1:
            return self.canvas
        return cv2.resize(
            self.canvas, (self.width, self.height), interpolation=cv2.INTER_AREA
        )


def _box_corners(center, half, rot) -> np.ndarray | None:
    if center is None or half is None:
        return None
    c = np.asarray(center, float)
    h = np.asarray(half, float)
    r = np.eye(3) if rot is None else np.asarray(rot, float)
    signs = np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float
    )
    return c + (signs * h) @ r.T


def _face_normal(quad: np.ndarray) -> np.ndarray:
    return norm(cross(quad[1] - quad[0], quad[3] - quad[0]))
