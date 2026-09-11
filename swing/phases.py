import numpy as np

from config import ACTIVE_PHASES, SwingType
from pose.detector import FramePose

# phase_name -> index into the frames list (not frame_idx)
PhaseMap = dict[str, int]

# Left wrist — leads the swing for a right-handed golfer
_WRIST = 15  

def _wrist_displacement(frames: list[FramePose]) -> np.ndarray:
    """Wrist distance from address position. Uses median of first 20 frames as
    reference so frame-0 jitter doesn't affect the baseline."""
    xs = np.array([f.landmarks[_WRIST].x for f in frames])
    ys = np.array([f.landmarks[_WRIST].y for f in frames])
    ref_n = min(20, len(frames) // 4)
    ref_x = np.median(xs[:ref_n])
    ref_y = np.median(ys[:ref_n])
    return np.sqrt((xs - ref_x)**2 + (ys - ref_y)**2)

def _last_visible_frame(frames: list[FramePose]) -> int:
    """Walk backward from the end to skip black/fade-out frames."""
    for i in range(len(frames) - 1, -1, -1):
        if frames[i].frame.mean() > 15:
            return i
    return len(frames) - 1

def _first_visible_frame(frames: list[FramePose]) -> int:
    """Walk forward from the start to skip black/fade-in frames."""
    for i in range(len(frames)):
        if frames[i].frame.mean() > 15:
            return i
    return 0

def _find_top(dist: np.ndarray) -> int:
    """Top of backswing: max displacement before the peak wrist speed.
    The downswing is always the fastest wrist motion — anchors the search."""
    vel = np.abs(np.gradient(dist))
    vel = np.convolve(vel, np.ones(7) / 7, mode='same')
    speed_peak = int(np.argmax(vel))
    return int(np.argmax(dist[:speed_peak + 1]))

def _find_swing_start(dist: np.ndarray, top_idx: int) -> int:
    """Walk backward from top to find where the backswing begins.
    Returns the last frame where displacement was still below 20% of peak."""
    threshold = dist[top_idx] * 0.20
    for i in range(top_idx, -1, -1):
        if dist[i] < threshold:
            return min(i + 1, top_idx)
    return 0

def _find_impact(frames: list[FramePose], top_idx: int) -> int:
    """Frame after top where wrist y is maximum — lowest physical point = impact zone."""
    ys = np.array([f.landmarks[_WRIST].y for f in frames])
    return top_idx + int(np.argmax(ys[top_idx:]))

def detect_phases(frames: list[FramePose], swing_type: SwingType = "full") -> PhaseMap:
    """Detect swing phase boundaries. Returns {phase_name: frame list index}."""
    n = _last_visible_frame(frames) + 1
    first = _first_visible_frame(frames)
    dist = _wrist_displacement(frames)
    top = _find_top(dist)
    move = _find_swing_start(dist, top)
    imp = _find_impact(frames, top)

    bs_quarter = max((top - move) // 4, 1)
    anchors = {
        "setup": first,
        "address": max(move - 5, first),
        "takeaway": move + bs_quarter, # 25% into backswing
        "backswing": move + bs_quarter * 2, # 50% into backswing
        "top": top,
        "downswing": top + max((imp - top) // 2, 1),
        "impact": imp,
        "follow_through": imp + (n - 1 - imp) // 2,
        "finish": n - 1,
    }

    return {phase: anchors[phase] for phase in ACTIVE_PHASES[swing_type]}
