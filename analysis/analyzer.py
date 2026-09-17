from dataclasses import dataclass
from typing import Literal

from config import IDEAL_RANGES, Club, SwingType
from swing.metrics import Metrics
from swing.rotation import rotation_from_span

CameraAngle = Literal["dtl", "face_on", "unknown"]

# Rotation metrics require face-on view; geometry metrics work from DTL.
_DTL_METRICS    = {"spine_angle", "knee_flex", "head_drift_lateral", "head_drift_vertical"}
_FACEON_METRICS = {"hip_rotation", "shoulder_rotation", "hip_shoulder_separation", "hip_sway"}

@dataclass
class Issue:
    phase: str
    metric: str
    measured: float
    ideal_min: float
    ideal_max: float
    severity: str # "warn" or "error"

def _get(m: Metrics, key: str, address_m: Metrics) -> float | None:
    """Extract the metric value for a given IDEAL_RANGES key.
    Drift metrics are computed relative to the address frame."""
    if key == "spine_angle":
        return m.spine_angle
    if key == "knee_flex":
        if m.left_knee_flex is None or m.right_knee_flex is None: return None
        return (m.left_knee_flex + m.right_knee_flex) / 2
    if key == "hip_rotation":
        return rotation_from_span(m.hip_span, address_m.hip_span)
    if key == "shoulder_rotation":
        return rotation_from_span(m.shoulder_span, address_m.shoulder_span)
    if key == "hip_shoulder_separation":
        hip_r = rotation_from_span(m.hip_span, address_m.hip_span)
        sho_r = rotation_from_span(m.shoulder_span, address_m.shoulder_span)
        if hip_r is None or sho_r is None: return None
        return abs(sho_r - hip_r)
    if key == "head_drift_lateral":
        if m.head_x is None or address_m.head_x is None: return None
        return abs(m.head_x - address_m.head_x)
    if key == "head_drift_vertical":
        if m.head_y is None or address_m.head_y is None: return None
        return abs(m.head_y - address_m.head_y)
    if key == "hip_sway":
        if m.hip_x is None or address_m.hip_x is None: return None
        return abs(m.hip_x - address_m.hip_x)
    return None

def analyze(
    phases: dict[str, int],
    metrics: list[Metrics],
    club: Club = "driver",
    swing_type: SwingType = "full",
    angle: CameraAngle = "unknown",
) -> list[Issue]:
    ranges     = IDEAL_RANGES[swing_type][club]
    address_m  = metrics[phases["address"]]
    issues: list[Issue] = []

    for phase, frame_idx in phases.items():
        m = metrics[frame_idx]
        for metric_key, phase_ranges in ranges.items():
            if angle == "dtl" and metric_key in _FACEON_METRICS:
                continue
            if angle == "face_on" and metric_key in _DTL_METRICS:
                continue
            ideal = phase_ranges.get(phase)
            if ideal is None:
                continue
            value = _get(m, metric_key, address_m)
            if value is None:
                continue
            lo, hi = ideal
            if not (lo <= value <= hi):
                deviation = min(abs(value - lo), abs(value - hi))
                severity  = "error" if deviation > (hi - lo) * 0.5 else "warn"
                issues.append(Issue(phase, metric_key, value, lo, hi, severity))

    return issues
