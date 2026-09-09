from typing import Literal

# Swing types based on the shot.
SwingType = Literal["full", "partial", "pitch", "chip"]

# Phases active per swing type. Phase detection is skipped for inactive phases.
ACTIVE_PHASES: dict[SwingType, list[str]] = {
    "full": ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"],
    "partial": ["address", "takeaway", "backswing", "top", "downswing", "impact", "follow_through", "finish"],
    "pitch": ["address", "backswing", "impact", "follow_through", "finish"],
    "chip": ["address", "impact", "finish"],
}

# MediaPipe settings
POSE_MODEL_COMPLEXITY = 1 # 0=lite, 1=full, 2=heavy
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# Savitzky-Golay landmark smoothing
SMOOTHING_WINDOW = 7 # must be odd
SMOOTHING_POLYORDER = 2
