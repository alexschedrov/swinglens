from dataclasses import dataclass

from config import LM_HEAD, LM_L_ELBOW, LM_L_HIP, LM_L_SHOULDER, LM_L_WRIST, LM_R_HIP, LM_R_SHOULDER
from pose.detector import FramePose
from swing.metrics import Metrics, _angle_at, _pt
from swing.phases import PhaseMap, phase_at_frame
from swing.rotation import rotation_from_span

@dataclass
class AngularVelocities:
    pelvis: float | None # degrees/sec, finite-difference of pelvis_rotation
    torso: float | None
    lead_arm: float | None

@dataclass
class SwingState:
    frame_idx: int
    timestamp: float
    phase: str | None # None before the first detected phase
    pelvis_rotation: float | None # degrees from address (acos-trick)
    torso_rotation: float | None
    lead_arm_angle: float | None # degrees at the lead elbow
    wrist_position: tuple[float, float] | None # normalized (x, y), lead wrist
    head_position: tuple[float, float] | None
    com_proxy: tuple[float, float] | None # weighted hip/shoulder/head blend, not a real COM
    angular_velocities: AngularVelocities

def _lead_arm_angle(fp: FramePose) -> float | None:
    lms = fp.landmarks
    if any(lms[i].visibility <= 0.4 for i in (LM_L_SHOULDER, LM_L_ELBOW, LM_L_WRIST)):
        return None
    h, w = fp.frame.shape[:2]
    return _angle_at(_pt(lms[LM_L_SHOULDER], h, w), _pt(lms[LM_L_ELBOW], h, w), _pt(lms[LM_L_WRIST], h, w))


def _wrist_position(fp: FramePose) -> tuple[float, float] | None:
    lm = fp.landmarks[LM_L_WRIST]
    return (lm.x, lm.y) if lm.visibility > 0.4 else None


def _com_proxy(fp: FramePose) -> tuple[float, float] | None:
    """Weighted blend of hip/shoulder/head midpoints. A rough proxy, not a real
    center-of-mass model -- an actual COM needs segment masses we don't have."""
    lms = fp.landmarks
    needed = (LM_L_HIP, LM_R_HIP, LM_L_SHOULDER, LM_R_SHOULDER, LM_HEAD)
    if any(lms[i].visibility <= 0.4 for i in needed):
        return None
    hip_x = (lms[LM_L_HIP].x + lms[LM_R_HIP].x) / 2
    hip_y = (lms[LM_L_HIP].y + lms[LM_R_HIP].y) / 2
    sho_x = (lms[LM_L_SHOULDER].x + lms[LM_R_SHOULDER].x) / 2
    sho_y = (lms[LM_L_SHOULDER].y + lms[LM_R_SHOULDER].y) / 2
    x = 0.5 * hip_x + 0.3 * sho_x + 0.2 * lms[LM_HEAD].x
    y = 0.5 * hip_y + 0.3 * sho_y + 0.2 * lms[LM_HEAD].y
    return (x, y)

def _rate(prev: float | None, curr: float | None, dt: float) -> float | None:
    if prev is None or curr is None:
        return None
    return (curr - prev) / dt

def compute_all(frames: list[FramePose], metrics: list[Metrics], phases: PhaseMap) -> list[SwingState]:
    """Build the per-frame state trajectory S = [s_0, ..., s_T]. Unlike Metrics (a raw
    per-frame measurement), SwingState adds phase labels and finite-difference angular
    velocities, both of which need the whole sequence rather than a single frame."""
    address_m = metrics[phases["address"]]

    # positions[i] = (pelvis_rotation, torso_rotation, lead_arm_angle), used below both
    # as SwingState fields and as the raw series angular_velocities differences over.
    positions = []
    for frame, m in zip(frames, metrics):
        positions.append((
            rotation_from_span(m.hip_span, address_m.hip_span),
            rotation_from_span(m.shoulder_span, address_m.shoulder_span),
            _lead_arm_angle(frame),
        ))

    states = []
    for i, (frame, m) in enumerate(zip(frames, metrics)):
        pelvis_rotation, torso_rotation, lead_arm_angle = positions[i]

        angular_velocities = AngularVelocities(None, None, None)
        if i > 0:
            dt = frame.timestamp - frames[i - 1].timestamp
            if dt > 0:
                prev_pelvis, prev_torso, prev_lead_arm = positions[i - 1]
                angular_velocities = AngularVelocities(
                    pelvis=_rate(prev_pelvis, pelvis_rotation, dt),
                    torso=_rate(prev_torso, torso_rotation, dt),
                    lead_arm=_rate(prev_lead_arm, lead_arm_angle, dt),
                )

        states.append(SwingState(
            frame_idx=frame.frame_idx,
            timestamp=frame.timestamp,
            phase=phase_at_frame(phases, i),
            pelvis_rotation=pelvis_rotation,
            torso_rotation=torso_rotation,
            lead_arm_angle=lead_arm_angle,
            wrist_position=_wrist_position(frame),
            head_position=(m.head_x, m.head_y) if m.head_x is not None else None,
            com_proxy=_com_proxy(frame),
            angular_velocities=angular_velocities,
        ))

    return states
