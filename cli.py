import argparse
from pathlib import Path

import cv2
import numpy as np

from pose.detector import FramePose, detect
from pose.smoother import smooth
from swing.phases import detect_phases

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

def _draw_skeleton(img: np.ndarray, lms: list) -> None:
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

def _phase_at_frame(phases: dict[str, int], i: int) -> str | None:
    """Current phase label for frame index i: the last phase anchor at or before i.
    None means frame i comes before the first detected phase (e.g. pre-swing setup)."""
    label = None
    for phase, idx in phases.items():
        if idx <= i:
            label = phase
    return label

def _export_annotated_video(frames: list[FramePose], phases: dict[str, int], out_path: str, fps: float) -> None:
    h, w = frames[0].frame.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i, fp in enumerate(frames):
        img = fp.frame.copy()
        _draw_skeleton(img, fp.landmarks)
        phase = _phase_at_frame(phases, i) or "..."
        label = f"{phase}   t={fp.timestamp:.2f}s"
        cv2.putText(img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()

def run(video: str, swing_type: str, output: str) -> None:
    print(f"[1/3] detecting pose: {video}")
    frames = detect(video)

    print(f"[2/3] smoothing landmarks ({len(frames)} frames)")
    frames = smooth(frames)

    print("[3/3] detecting swing phases")
    phases = detect_phases(frames, swing_type=swing_type)
    for phase, idx in phases.items():
        print(f"  {phase:<16} frame {frames[idx].frame_idx:>4}  t={frames[idx].timestamp:.2f}s")

    fps = len(frames) / frames[-1].timestamp if frames[-1].timestamp else 30.0
    _export_annotated_video(frames, phases, output, fps)
    print(f"saved annotated video -> {output}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Golf swing posture tracker")
    parser.add_argument("--video", required=True, help="Path to swing video")
    parser.add_argument("--swing-type", default="full", choices=["full", "partial", "pitch", "chip"])
    parser.add_argument("--output", help="Path for the annotated output video")
    args = parser.parse_args()

    output = args.output or str(Path(args.video).with_stem(Path(args.video).stem + "_annotated"))
    run(args.video, args.swing_type, output)

if __name__ == "__main__":
    main()
