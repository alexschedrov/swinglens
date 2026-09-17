import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from analysis.analyzer import _DTL_METRICS, _FACEON_METRICS, _get
from config import IDEAL_RANGES, LM_L_HIP, LM_L_SHOULDER, LM_R_HIP, LM_R_SHOULDER
from pose.detector import FramePose
from swing.metrics import Metrics
from swing.phases import PhaseMap, phase_at_frame
from swing.state import SwingState

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

def plot_rotation_comparison(rows: list[tuple], output: str) -> None:
    """rows: list of (phase, acos_hip, z_hip, acos_sho, z_sho)."""
    phase_labels = [r[0] for r in rows]
    x = range(len(rows))

    fig, (ax_hip, ax_sho) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax_hip.plot(x, [r[1] for r in rows], "o-", label="acos-trick")
    ax_hip.plot(x, [r[2] for r in rows], "o-", label="z-derived")
    ax_hip.set_title("Hip rotation")
    ax_hip.set_ylabel("degrees from address")

    ax_sho.plot(x, [r[3] for r in rows], "o-", label="acos-trick")
    ax_sho.plot(x, [r[4] for r in rows], "o-", label="z-derived")
    ax_sho.set_title("Shoulder rotation")

    for ax in (ax_hip, ax_sho):
        ax.set_xticks(list(x))
        ax.set_xticklabels(phase_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("acos-trick vs. z-derived rotation (self-consistency check)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

def plot_birdseye_rotation(frames: list[FramePose], phases: PhaseMap, output: str) -> None:
    """Bird's-eye view (x/z plane, y dropped): the hip and shoulder lines at each phase,
    centered on their own midpoint so only rotation shows (translation is removed). This
    is a direct picture of what acos(span / address_span) is measuring -- the line's
    x-width shrinking as it rotates in the x/z plane."""
    fig, (ax_hip, ax_sho) = plt.subplots(1, 2, figsize=(11, 5.5))
    cmap = plt.get_cmap("viridis")
    n = len(phases)

    for ax, (a_idx, b_idx), title in (
        (ax_hip, (LM_L_HIP, LM_R_HIP), "Hip line"),
        (ax_sho, (LM_L_SHOULDER, LM_R_SHOULDER), "Shoulder line"),
    ):
        for i, (phase, idx) in enumerate(phases.items()):
            a, b = frames[idx].landmarks[a_idx], frames[idx].landmarks[b_idx]
            cx, cz = (a.x + b.x) / 2, (a.z + b.z) / 2
            color = cmap(i / max(n - 1, 1))
            ax.plot([a.x - cx, b.x - cx], [a.z - cz, b.z - cz], "-o", color=color, label=phase)
        ax.set_title(title)
        ax.set_xlabel("x (lateral, line-centered)")
        ax.set_ylabel("z (depth, mediapipe units)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    fig.suptitle("Bird's-eye view: hip/shoulder line rotating through the swing")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

def plot_birdseye_filmstrip(frames: list[FramePose], phases: PhaseMap, output: str) -> None:
    """Camera frame next to its bird's-eye (x/z) shoulder-line reprojection, one column
    per phase -- a static filmstrip standing in for the animated side-by-side."""
    n = len(phases)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))

    for col, (phase, idx) in enumerate(phases.items()):
        frame = frames[idx]
        img_ax, bev_ax = axes[0, col], axes[1, col]

        rgb = cv2.cvtColor(frame.frame, cv2.COLOR_BGR2RGB)
        img_ax.imshow(rgb)
        img_ax.set_title(phase, fontsize=8)
        img_ax.axis("off")

        a, b = frame.landmarks[LM_L_SHOULDER], frame.landmarks[LM_R_SHOULDER]
        cx, cz = (a.x + b.x) / 2, (a.z + b.z) / 2
        bev_ax.plot([a.x - cx, b.x - cx], [a.z - cz, b.z - cz], "-o", color="tab:orange")
        bev_ax.set_xlim(-0.25, 0.25)
        bev_ax.set_ylim(-0.25, 0.25)
        bev_ax.set_aspect("equal")
        bev_ax.set_xticks([])
        bev_ax.set_yticks([])
        bev_ax.grid(alpha=0.3)
        if col == 0:
            bev_ax.set_ylabel("bird's-eye\n(shoulder line)", fontsize=8)

    fig.suptitle("Camera frame vs. bird's-eye x/z reprojection, through the swing")
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)

def plot_metrics_timeseries(frames: list[FramePose], metrics: list[Metrics], phases: PhaseMap, output: str) -> None:
    ts = [f.timestamp for f in frames]

    fig, axes = plt.subplots(5, 1, figsize=(9, 13), sharex=True)

    axes[0].plot(ts, [m.spine_angle for m in metrics])
    axes[0].set_title("Spine angle (deg)")

    axes[1].plot(ts, [m.left_knee_flex for m in metrics], label="left")
    axes[1].plot(ts, [m.right_knee_flex for m in metrics], label="right")
    axes[1].set_title("Knee flex (deg)")
    axes[1].legend(fontsize=8)

    axes[2].plot(ts, [m.hip_span for m in metrics], label="hip")
    axes[2].plot(ts, [m.shoulder_span for m in metrics], label="shoulder")
    axes[2].set_title("Span (normalized line width)")
    axes[2].legend(fontsize=8)

    axes[3].plot(ts, [m.head_x for m in metrics], label="x")
    axes[3].plot(ts, [m.head_y for m in metrics], label="y")
    axes[3].set_title("Head position (normalized)")
    axes[3].legend(fontsize=8)

    axes[4].plot(ts, [m.hip_x for m in metrics])
    axes[4].set_title("Hip lateral position (sway proxy)")
    axes[4].set_xlabel("time (s)")

    for i, ax in enumerate(axes):
        for phase, idx in phases.items():
            ax.axvline(frames[idx].timestamp, color="gray", ls="--", lw=0.6, alpha=0.6)
            if i == 0:
                ax.text(frames[idx].timestamp, ax.get_ylim()[1], phase, fontsize=6, rotation=90, va="top")
        ax.grid(alpha=0.3)

    fig.suptitle("Metrics over time (dashed lines = phase boundaries)")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

def plot_state_timeseries(states: list[SwingState], phases: PhaseMap, output: str) -> None:
    ts = [s.timestamp for s in states]

    fig, (ax_rot, ax_vel) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_rot.plot(ts, [s.pelvis_rotation for s in states], label="pelvis")
    ax_rot.plot(ts, [s.torso_rotation for s in states], label="torso")
    ax_rot.plot(ts, [s.lead_arm_angle for s in states], label="lead arm angle")
    ax_rot.set_ylabel("degrees")
    ax_rot.set_title("Rotation / angle state")
    ax_rot.legend(fontsize=8)

    ax_vel.plot(ts, [s.angular_velocities.pelvis for s in states], label="pelvis")
    ax_vel.plot(ts, [s.angular_velocities.torso for s in states], label="torso")
    ax_vel.plot(ts, [s.angular_velocities.lead_arm for s in states], label="lead arm")
    ax_vel.axhline(0, color="gray", lw=0.5)
    ax_vel.set_ylabel("degrees/sec")
    ax_vel.set_xlabel("time (s)")
    ax_vel.set_title("Angular velocities (finite difference)")
    ax_vel.legend(fontsize=8)

    for ax in (ax_rot, ax_vel):
        for idx in phases.values():
            ax.axvline(states[idx].timestamp, color="gray", ls="--", lw=0.6, alpha=0.5)
        ax.grid(alpha=0.3)

    fig.suptitle("SwingState: rotation and angular velocity over time")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

def plot_state_trajectories(states: list[SwingState], output: str) -> None:
    fig, (ax_wrist, ax_com) = plt.subplots(1, 2, figsize=(11, 5))

    for ax, field, title in (
        (ax_wrist, "wrist_position", "Lead wrist path"),
        (ax_com, "com_proxy", "COM proxy path"),
    ):
        pts = [(s.timestamp, getattr(s, field)) for s in states if getattr(s, field) is not None]
        if not pts:
            ax.set_title(f"{title} (no data)")
            continue
        times = [t for t, _ in pts]
        xs = [p[0] for _, p in pts]
        ys = [p[1] for _, p in pts]
        sc = ax.scatter(xs, ys, c=times, cmap="viridis", s=8)
        ax.plot(xs, ys, color="gray", lw=0.5, alpha=0.4)
        ax.invert_yaxis() # image-space y grows downward
        ax.set_title(title)
        ax.set_xlabel("x (normalized)")
        ax.set_ylabel("y (normalized)")
        fig.colorbar(sc, ax=ax, label="time (s)")

    fig.suptitle("SwingState trajectories")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

def plot_issues_grid(
    phases: PhaseMap,
    metrics: list[Metrics],
    club: str,
    swing_type: str,
    angle: str,
    output: str,
) -> None:
    """Status grid: every (metric, phase) cell IDEAL_RANGES defines, colored by whether
    the measured value passes, warns, or errors against that range. Reuses analyzer._get
    so this grid always matches analyze()'s own pass/fail logic."""
    ranges = IDEAL_RANGES[swing_type][club]
    address_m = metrics[phases["address"]]
    metric_keys = list(ranges.keys())
    phase_names = list(phases.keys())

    status = np.zeros((len(metric_keys), len(phase_names))) # 0=n/a 1=pass 2=warn 3=error
    for i, metric_key in enumerate(metric_keys):
        if angle == "dtl" and metric_key in _FACEON_METRICS: continue
        if angle == "face_on" and metric_key in _DTL_METRICS: continue
        for j, phase in enumerate(phase_names):
            ideal = ranges[metric_key].get(phase)
            if ideal is None:
                continue
            value = _get(metrics[phases[phase]], metric_key, address_m)
            if value is None:
                continue
            lo, hi = ideal
            if lo <= value <= hi:
                status[i, j] = 1
            else:
                deviation = min(abs(value - lo), abs(value - hi))
                status[i, j] = 3 if deviation > (hi - lo) * 0.5 else 2

    cmap = mcolors.ListedColormap(["#e0e0e0", "#4caf50", "#ff9800", "#f44336"])
    fig, ax = plt.subplots(figsize=(1.1 * len(phase_names) + 2, 0.5 * len(metric_keys) + 2))
    ax.imshow(status, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(phase_names)))
    ax.set_xticklabels(phase_names, rotation=45, ha="right")
    ax.set_yticks(range(len(metric_keys)))
    ax.set_yticklabels(metric_keys)
    ax.set_title(
        f"Issue grid -- club={club} swing_type={swing_type} angle={angle}\n"
        "gray=n/a  green=pass  orange=warn  red=error"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
