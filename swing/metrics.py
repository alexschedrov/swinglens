import math
from dataclasses import dataclass

import numpy as np

from pose.detector import FramePose

@dataclass
class Metrics:
    spine_angle: float | None # degrees tilt from vertical (hip-shoulder)
    left_knee_flex: float | None # degrees bent from straight (0 = straight)
    right_knee_flex: float | None
    hip_span: float | None # normalized x-width of hip line (0-1)
    shoulder_span: float | None # normalized x-width of shoulder line (0-1)
    hip_shoulder_separation: float | None # placeholder; computed in analyzer from spans
    head_x: float | None # normalized lateral head position
    head_y: float | None # normalized vertical head position
    hip_x: float | None # normalized lateral hip midpoint (sway proxy)

def _pt(lm, h: int, w: int) -> np.ndarray:
    return np.array([lm.x * w, lm.y * h], dtype=float)

def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex b in the a-b-c triplet."""
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))

def _tilt_from_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return math.degrees(math.atan2(abs(dx), abs(dy)))

def compute(fp: FramePose) -> Metrics:
    lms = fp.landmarks
    h, w = fp.frame.shape[:2]
    def pt(i):   return _pt(lms[i], h, w)
    def ok(*ii): return all(lms[i].visibility > 0.4 for i in ii)

    spine_angle = None
    if ok(11, 12, 23, 24):
        mid_hip = (pt(23) + pt(24)) / 2
        mid_sho = (pt(11) + pt(12)) / 2
        spine_angle = _tilt_from_vertical(mid_hip, mid_sho)

    left_knee_flex  = (180 - _angle_at(pt(23), pt(25), pt(27))) if ok(23, 25, 27) else None
    right_knee_flex = (180 - _angle_at(pt(24), pt(26), pt(28))) if ok(24, 26, 28) else None

    # Span = normalized x-distance between landmarks; compresses as the golfer rotates.
    # Rotation angle is derived from this via swing/rotation.py's weak-perspective trick.
    hip_span      = abs(lms[24].x - lms[23].x) if ok(23, 24) else None
    shoulder_span = abs(lms[12].x - lms[11].x) if ok(11, 12) else None

    hip_shoulder_separation = None # computed in analyzer once rotation angles are known

    head_x = lms[0].x if ok(0) else None
    head_y = lms[0].y if ok(0) else None
    hip_x  = (lms[23].x + lms[24].x) / 2 if ok(23, 24) else None

    return Metrics(
        spine_angle=spine_angle,
        left_knee_flex=left_knee_flex,
        right_knee_flex=right_knee_flex,
        hip_span=hip_span,
        shoulder_span=shoulder_span,
        hip_shoulder_separation=hip_shoulder_separation,
        head_x=head_x,
        head_y=head_y,
        hip_x=hip_x,
    )

def compute_all(frames: list[FramePose]) -> list[Metrics]:
    return [compute(fp) for fp in frames]
