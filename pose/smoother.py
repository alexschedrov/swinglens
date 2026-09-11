import numpy as np
from scipy.signal import savgol_filter

from config import SMOOTHING_WINDOW, SMOOTHING_POLYORDER
from pose.detector import FramePose, Landmark

def smooth(frames: list[FramePose]) -> list[FramePose]:
    """Apply Savitzky-Golay smoothing to landmark x/y/z across all frames.
    Returns a new list of FramePose with smoothed coordinates; visibility is unchanged."""
    if len(frames) < SMOOTHING_WINDOW:
        return frames  # too short to smooth

    num_landmarks = len(frames[0].landmarks)

    # Build arrays of shape (n, num_landmarks) for each coordinate
    xs = np.array([[lm.x for lm in f.landmarks] for f in frames])  # (n, 33)
    ys = np.array([[lm.y for lm in f.landmarks] for f in frames])  # (n, 33)
    zs = np.array([[lm.z for lm in f.landmarks] for f in frames])  # (n, 33)

    # Smooth each landmark's trajectory independently along the time axis
    xs_s = savgol_filter(xs, SMOOTHING_WINDOW, SMOOTHING_POLYORDER, axis=0)
    ys_s = savgol_filter(ys, SMOOTHING_WINDOW, SMOOTHING_POLYORDER, axis=0)
    zs_s = savgol_filter(zs, SMOOTHING_WINDOW, SMOOTHING_POLYORDER, axis=0)

    smoothed = []
    for i, frame in enumerate(frames):
        landmarks = [
            Landmark(xs_s[i, j], ys_s[i, j], zs_s[i, j], frame.landmarks[j].visibility)
            for j in range(num_landmarks)
        ]
        smoothed.append(FramePose(frame.frame_idx, frame.timestamp, landmarks, frame.frame))

    return smoothed
