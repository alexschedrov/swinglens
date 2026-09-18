import cv2
import numpy as np

from pose.detector import FramePose
from swing.phases import PhaseMap, phase_at_frame

# Skeleton edges to draw, grouped by color. Indices are MediaPipe Pose landmarks.
# Covers all 33 landmarks: face (0-10), torso (11,12,23,24), arms + hands (13-22),
# legs + feet (25-32).
_SEGMENTS = {
    (0, 1): (150, 150, 150), # face
    (1, 2): (150, 150, 150),
    (2, 3): (150, 150, 150),
    (3, 7): (150, 150, 150),
    (0, 4): (150, 150, 150),
    (4, 5): (150, 150, 150),
    (5, 6): (150, 150, 150),
    (6, 8): (150, 150, 150),
    (9, 10): (150, 150, 150),
    (11, 12): (200, 200, 200), # shoulders
    (11, 23): (50, 220, 50), # L shoulder -> L hip
    (12, 24): (50, 220, 50), # R shoulder -> R hip
    (23, 24): (50, 220, 50), # hips
    (11, 13): (230, 160, 30), # L shoulder -> L elbow
    (13, 15): (230, 160, 30), # L elbow -> L wrist
    (12, 14): (230, 160, 30), # R shoulder -> R elbow
    (14, 16): (230, 160, 30), # R elbow -> R wrist
    (15, 17): (230, 190, 80), # L wrist -> L pinky
    (15, 19): (230, 190, 80), # L wrist -> L index
    (15, 21): (230, 190, 80), # L wrist -> L thumb
    (17, 19): (230, 190, 80), # L pinky -> L index
    (16, 18): (230, 190, 80), # R wrist -> R pinky
    (16, 20): (230, 190, 80), # R wrist -> R index
    (16, 22): (230, 190, 80), # R wrist -> R thumb
    (18, 20): (230, 190, 80), # R pinky -> R index
    (23, 25): (50, 50, 220), # L hip -> L knee
    (25, 27): (50, 50, 220), # L knee -> L ankle
    (24, 26): (50, 50, 220), # R hip -> R knee
    (26, 28): (50, 50, 220), # R knee -> R ankle
    (27, 29): (50, 110, 220), # L ankle -> L heel
    (27, 31): (50, 110, 220), # L ankle -> L foot index
    (29, 31): (50, 110, 220), # L heel -> L foot index
    (28, 30): (50, 110, 220), # R ankle -> R heel
    (28, 32): (50, 110, 220), # R ankle -> R foot index
    (30, 32): (50, 110, 220), # R heel -> R foot index
}

def draw_skeleton(img: np.ndarray, lms: list) -> None:
    h, w = img.shape[:2]
    for (a, b), color in _SEGMENTS.items():
        if lms[a].visibility > 0.4 and lms[b].visibility > 0.4:
            pa = int(lms[a].x * w), int(lms[a].y * h)
            pb = int(lms[b].x * w), int(lms[b].y * h)
            cv2.line(img, pa, pb, color, 2, cv2.LINE_AA)

    for lm in lms:
        if lm.visibility > 0.4:
            color = (0, 255, 255) if lm.visibility > 0.7 else (0, 165, 255)
            cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 3, color, -1, cv2.LINE_AA)

def export_annotated_video(frames: list[FramePose], phases: PhaseMap, out_path: str, fps: float) -> None:
    h, w = frames[0].frame.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, fp in enumerate(frames):
        img = fp.frame.copy()
        draw_skeleton(img, fp.landmarks)
        phase = phase_at_frame(phases, i) or "..."
        label = f"{phase}   t={fp.timestamp:.2f}s"
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()
