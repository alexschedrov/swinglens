from dataclasses import dataclass
from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

import config

# MediaPipe hosts one pretrained pose model per complexity level. We 
# download whichever one config.py asks for and cache it under models folder.
_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}
_MODEL_NAMES = {0: "pose_landmarker_lite.task", 1: "pose_landmarker_full.task", 2: "pose_landmarker_heavy.task"}
_MODELS_DIR = Path(__file__).parent.parent / "models"

def _ensure_model(complexity: int) -> str:
    """Download the pose model for this complexity level if not already cached."""
    _MODELS_DIR.mkdir(exist_ok=True)
    path = _MODELS_DIR / _MODEL_NAMES[complexity]
    if not path.exists():
        url = _MODEL_URLS[complexity]
        print(f"  downloading model: {url}")
        urllib.request.urlretrieve(url, path)
    return str(path)

@dataclass
class Landmark:
    x: float # normalized [0, 1] horizontal position in frame
    y: float # normalized [0, 1] vertical position in frame
    z: float # depth relative to hip midpoint (mediapipe units)
    visibility: float # confidence [0, 1]

@dataclass
class FramePose:
    frame_idx: int
    timestamp: float # seconds from video start
    landmarks: list[Landmark] # 33 items, indexed by PoseLandmark enum value
    frame: np.ndarray # raw BGR frame, kept for annotation

def detect(video_path: str) -> list[FramePose]:
    """Run MediaPipe Pose on every frame. Frames with no detected person are skipped."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results: list[FramePose] = []

    model_path = _ensure_model(config.POSE_MODEL_COMPLEXITY)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
    )

    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            timestamp_ms = int(frame_idx / fps * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            detection = landmarker.detect_for_video(mp_image, timestamp_ms)

            if detection.pose_landmarks:
                # First (only) person
                lms = detection.pose_landmarks[0]
                results.append(FramePose(
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    landmarks=[Landmark(lm.x, lm.y, lm.z, lm.visibility) for lm in lms],
                    frame=bgr.copy(),
                ))

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"  detecting: {frame_idx}/{total} frames", end="\r")

    cap.release()
    print(f"  detecting: done, {len(results)}/{total} frames with pose")
    return results
