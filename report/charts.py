import base64
import json
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from analysis.analyzer import _DTL_METRICS, _FACEON_METRICS, _get
from config import IDEAL_RANGES, LM_L_HIP, LM_L_SHOULDER, LM_R_HIP, LM_R_SHOULDER
from pose.detector import FramePose, Landmark
from swing.metrics import Metrics
from swing.phases import PhaseMap
from swing.rotation import RotationRow
from swing.state import SwingState
from visualization.visualization import _SEGMENTS, draw_skeleton

_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(Path(__file__).parent.parent / "templates"),
    autoescape=select_autoescape(default=False), # scrubber_controls.html embeds raw Plotly HTML/JS
)

# Single source of truth for every color, shared by the Python charts and
# scrubber_controls.html (via colors=_COLORS in the template render).
_COLORS = dict(
    text="#475569", title="#1e293b", joint="#333333", phase_line="gray",
    trajectory_line="rgba(150,150,150,0.4)",
    grid_na="#e0e0e0", grid_pass="#4caf50", grid_warn="#ff9800", grid_error="#f44336",
    accent="#4f46e5", accent_bg="#eef2ff", accent_border="#c7d2fe",
    accent_text="#4338ca", accent_dark="#3730a3",
)

_FONT = dict(
    family='-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    size=13, color=_COLORS["text"],
)
_TITLE_FONT = dict(family=_FONT["family"], size=16, color=_COLORS["title"])

def _title(text: str) -> dict:
    """Shorthand for the title=dict(text=..., font=_TITLE_FONT) pattern every chart repeats."""
    return dict(text=text, font=_TITLE_FONT)

_SKELETON_GROUPS: dict[str, list[tuple[int, int]]] = {}
for _pair, _bgr in _SEGMENTS.items():
    _SKELETON_GROUPS.setdefault(f"rgb({_bgr[2]},{_bgr[1]},{_bgr[0]})", []).append(_pair)
_SKELETON_GROUP_COLORS = list(_SKELETON_GROUPS.keys())

# MediaPipe's coordinate system is always camera-relative (x=image right, y=image
# down, z=depth away from camera). What those axes mean physically for the swing
# depends on which way the camera was actually pointed.
_AXIS_LABELS = {
    "dtl": ("x (down-the-line swing arc)", "y (vertical, up positive)", "z (depth ≈ lateral sway)"),
    "face_on": ("x (lateral, across target line)", "y (vertical, up positive)", "z (depth ≈ target line direction)"),
    "unknown": ("x (camera-horizontal)", "y (vertical, up positive)", "z (camera depth)"),
}

def _axis_labels(angle: str) -> tuple[str, str, str]:
    return _AXIS_LABELS.get(angle, _AXIS_LABELS["unknown"])

# DTL and face-on are roughly perpendicular camera angles, so one camera's estimated
# depth (z) axis is approximately the other camera's own lateral (x) axis -- the "side"
# panel in the scrubber isn't a generic side view, it's an estimate of the *other* angle.
_ANGLE_DISPLAY_NAME = {"dtl": "Down-the-line", "face_on": "Face-on"}
_COMPLEMENTARY_ANGLE = {"dtl": "face_on", "face_on": "dtl"}

def _scrubber_panel_titles(angle: str) -> tuple[str, str]:
    primary = _ANGLE_DISPLAY_NAME.get(angle, "Front")
    complement = _COMPLEMENTARY_ANGLE.get(angle)
    secondary = f"{_ANGLE_DISPLAY_NAME[complement]} view (estimated)" if complement else "Side view (estimated depth)"
    return f"{primary} view", secondary

def _add_phase_lines(fig: go.Figure, phases: PhaseMap, timestamps: list[float]) -> None:
    for idx in phases.values():
        fig.add_vline(x=timestamps[idx], line_width=1, line_dash="dash", line_color=_COLORS["phase_line"], opacity=0.5, row="all", col=1)

def _segment_xz(a: Landmark, b: Landmark) -> tuple[list[float], list[float]]:
    """A 2-point line segment (a, b) centered on its own midpoint, in the x/z (top-down,
    y dropped) plane -- the bird's-eye picture of what the acos-trick's span-shrinkage is
    measuring."""
    cx, cz = (a.x + b.x) / 2, (a.z + b.z) / 2
    return [a.x - cx, b.x - cx], [a.z - cz, b.z - cz]

def _skeleton_traces_2d(landmarks: list[Landmark], x_attr: str) -> list[go.Scatter]:
    """One 2D trace per bone-color group plus one joint-marker trace, in a fixed order so
    trace indices line up across every animation frame. x_attr picks which landmark
    coordinate is the plotted x-axis -- 'x' for a front view matching the camera, 'z' for
    a side view showing MediaPipe's estimated depth; y is always vertical (flipped so
    "up" is positive)."""
    bone_traces = []
    for color, pairs in _SKELETON_GROUPS.items():
        xs: list = []
        ys: list = []
        for a_idx, b_idx in pairs:
            a, b = landmarks[a_idx], landmarks[b_idx]
            if a.visibility <= 0.4 or b.visibility <= 0.4:
                continue
            xs += [getattr(a, x_attr), getattr(b, x_attr), None]
            ys += [-a.y, -b.y, None] # image y grows downward; flip so "up" is positive
        bone_traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=3), showlegend=False))

    visible = [lm for lm in landmarks if lm.visibility > 0.4]
    joints_trace = go.Scatter(
        x=[getattr(lm, x_attr) for lm in visible], y=[-lm.y for lm in visible],
        mode="markers", marker=dict(size=4, color=_COLORS["joint"]), showlegend=False,
    )
    return [*bone_traces, joints_trace]

def _thumbnail_data_uri(fp: FramePose, target_width: int = 320, quality: int = 65) -> str:
    """Skeleton-overlaid, downsized JPEG frame as a base64 data URI -- used to animate a
    'video' panel via Plotly's own frame/slider mechanism instead of a <video> element."""
    h, w = fp.frame.shape[:2]
    scale = target_width / w
    img = cv2.resize(fp.frame, (target_width, round(h * scale)))
    draw_skeleton(img, fp.landmarks)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

def rotation_comparison_figure(rows: list[RotationRow]) -> go.Figure:
    """Acos-trick vs. z-derived rotation at each phase, hip and shoulder side by side."""
    phase_labels = [r[0] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Hip rotation", "Shoulder rotation"))

    fig.add_trace(go.Scatter(x=phase_labels, y=[r[1] for r in rows], mode="lines+markers", name="acos-trick (hip)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=phase_labels, y=[r[2] for r in rows], mode="lines+markers", name="z-derived (hip)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=phase_labels, y=[r[3] for r in rows], mode="lines+markers", name="acos-trick (shoulder)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=phase_labels, y=[r[4] for r in rows], mode="lines+markers", name="z-derived (shoulder)"), row=1, col=2)

    fig.update_yaxes(title_text="degrees from address", row=1, col=1)
    fig.update_layout(title=_title("acos-trick vs. z-derived rotation (self-consistency check)"), font=_FONT, height=420, margin=dict(t=60, b=40))
    return fig

def metrics_timeseries_figure(frames: list[FramePose], metrics: list[Metrics], phases: PhaseMap) -> go.Figure:
    """Per-frame metrics over time, with a range slider on the shared time axis."""
    ts = [f.timestamp for f in frames]
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=("Spine angle (deg)", "Knee flex (deg)", "Span (normalized)", "Head position (normalized)", "Hip lateral position (sway proxy)"),
    )

    fig.add_trace(go.Scatter(x=ts, y=[m.spine_angle for m in metrics], name="spine angle", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[m.left_knee_flex for m in metrics], name="left knee"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[m.right_knee_flex for m in metrics], name="right knee"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[m.hip_span for m in metrics], name="hip span"), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[m.shoulder_span for m in metrics], name="shoulder span"), row=3, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[m.head_x for m in metrics], name="head x"), row=4, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[m.head_y for m in metrics], name="head y"), row=4, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[m.hip_x for m in metrics], name="hip x", showlegend=False), row=5, col=1)

    _add_phase_lines(fig, phases, ts)
    fig.update_xaxes(title_text="time (s)", rangeslider_visible=True, row=5, col=1)
    # Legend sits above the whole grid, not below: a bottom legend collided with row 5's
    # own tick labels and the rangeslider crowded beneath them.
    fig.update_layout(
        title=_title("Metrics over time (dashed lines = phase boundaries)"), font=_FONT,
        height=1200, margin=dict(t=100, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

def state_timeseries_figure(states: list[SwingState], phases: PhaseMap) -> go.Figure:
    """SwingState rotation and angular velocity over time, with a range slider on the
    shared time axis."""
    ts = [s.timestamp for s in states]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
                         subplot_titles=("Rotation / angle state", "Angular velocities (finite difference)"))

    fig.add_trace(go.Scatter(x=ts, y=[s.pelvis_rotation for s in states], name="pelvis"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[s.torso_rotation for s in states], name="torso"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[s.lead_arm_angle for s in states], name="lead arm angle"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[s.angular_velocities.pelvis for s in states], name="pelvis vel", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[s.angular_velocities.torso for s in states], name="torso vel", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts, y=[s.angular_velocities.lead_arm for s in states], name="lead arm vel", showlegend=False), row=2, col=1)

    _add_phase_lines(fig, phases, ts)
    fig.update_yaxes(title_text="degrees", row=1, col=1)
    fig.update_yaxes(title_text="degrees/sec", row=2, col=1)
    fig.update_xaxes(title_text="time (s)", rangeslider_visible=True, row=2, col=1)
    fig.update_layout(title=_title("SwingState: rotation and angular velocity over time"), font=_FONT, height=750, margin=dict(t=60, b=40))
    return fig

def state_trajectories_figure(states: list[SwingState], angle: str) -> go.Figure:
    """Lead-wrist and COM-proxy paths in the camera's own (x, y) view, colored by time.
    Depth (z) is dropped -- this chart is about the shape of the movement path as the
    camera actually saw it, not depth estimation (see the scrubber's estimated-angle
    panel for that)."""
    x_label, y_label, _ = _axis_labels(angle)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Lead wrist path", "COM proxy path"))

    for col, field in ((1, "wrist_position"), (2, "com_proxy")):
        pts = [(s.timestamp, getattr(s, field)) for s in states if getattr(s, field) is not None]
        if not pts:
            continue
        times = [t for t, _ in pts]
        xs = [p[0] for _, p in pts]
        ys = [-p[1] for _, p in pts] # image y grows downward; flip so "up" is positive
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers+lines",
                marker=dict(size=6, color=times, colorscale="Viridis", showscale=(col == 2), colorbar=dict(title="time (s)") if col == 2 else None),
                line=dict(color=_COLORS["trajectory_line"], width=1),
                hovertemplate="t=%{customdata:.2f}s<extra></extra>",
                customdata=times,
                showlegend=False,
            ),
            row=1, col=col,
        )
        fig.update_xaxes(title_text=x_label, row=1, col=col)

    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_layout(
        title=_title("SwingState trajectories"),
        font=_FONT, height=480, margin=dict(t=60, b=40),
    )
    return fig

def birdseye_rotation_figure(frames: list[FramePose], phases: PhaseMap, angle: str) -> go.Figure:
    """Hip and shoulder lines at each phase, in the top-down x/z plane (y dropped) -- the
    most direct picture of what the acos-trick's span-shrinkage is measuring. Equal aspect
    (scaleanchor) is locked here, unlike the scrubber's estimated-angle panel: the whole
    point of this chart is seeing the line's true shrinkage in x relative to its
    depth (z), so distorting that ratio would misrepresent the exact thing it's showing."""
    _, _, z_label = _axis_labels(angle)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Hip line", "Shoulder line"))

    n = len(phases)
    colors = sample_colorscale("Viridis", [i / max(n - 1, 1) for i in range(n)])

    for col, (a_idx, b_idx) in ((1, (LM_L_HIP, LM_R_HIP)), (2, (LM_L_SHOULDER, LM_R_SHOULDER))):
        for i, (phase, idx) in enumerate(phases.items()):
            xs, zs = _segment_xz(frames[idx].landmarks[a_idx], frames[idx].landmarks[b_idx])
            fig.add_trace(
                go.Scatter(
                    x=xs, y=zs,
                    mode="lines+markers", line=dict(color=colors[i], width=4), marker=dict(size=6, color=colors[i]),
                    name=phase, legendgroup=phase, showlegend=(col == 1),
                ),
                row=1, col=col,
            )
        fig.update_xaxes(title_text="x (lateral, line-centered)", row=1, col=col)

    fig.update_yaxes(title_text=z_label, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(title_text=z_label, scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(
        title=_title("Bird's-eye view: hip/shoulder line rotating through the swing"),
        font=_FONT, height=500, margin=dict(t=60, b=40),
    )
    return fig

def _global_axis_ranges(frames: list[FramePose], margin_frac: float = 0.15) -> dict:
    """Fixed axis ranges spanning every frame's visible landmarks. Without this, Plotly
    re-autoranges each panel to whatever subset of points is visible in the current
    frame, making the skeleton appear to zoom in and out as the slider moves instead of
    holding a steady scale.

    Uses the 1st-99th percentile, not the strict min/max: MediaPipe's z estimate has
    enough outlier frames that the raw range is dominated by a handful of bad ones,
    stretching the whole panel to fit them."""
    xs, ys, zs = [], [], []
    for fp in frames:
        for lm in fp.landmarks:
            if lm.visibility > 0.4:
                xs.append(lm.x)
                ys.append(-lm.y)
                zs.append(lm.z)

    def axis_range(vals: list[float]) -> list[float]:
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
        pad = (hi - lo) * margin_frac or 0.05
        return [lo - pad, hi + pad]

    return dict(
        xaxis=dict(range=axis_range(xs), autorange=False),
        yaxis=dict(range=axis_range(ys), autorange=False),
        zaxis=dict(range=axis_range(zs), autorange=False),
    )

def video_scrubber_html(frames: list[FramePose], angle: str, config: dict) -> str:
    """Synced three-panel scrubber, returned as ready-to-embed HTML (not a single
    go.Figure): the skeleton-overlaid video frame, a flat view of the skeleton matching
    the video's own camera angle (x/y), and a flat view estimating the *other* camera
    angle (z/y).

    These are three separate Plotly figures/divs, not subplots of one figure: go.Image
    traces require redraw=True to update, but redrawing an image trace alongside other
    "xy"-type subplots in the same figure visibly blanks it before each new frame paints.
    Separate Plotly instances avoid this since each gets its own independent redraw call.

    Since three figures means three built-in sliders would be redundant, one shared plain
    HTML range input + play/pause button drives all three via Plotly.animate calls in the
    appended <script> (see templates/scrubber_controls.html).

    DTL and face-on are roughly perpendicular camera angles, so one camera's estimated
    depth (z) is approximately the other camera's own lateral (x) axis -- e.g. for a DTL
    video, the second panel is labeled "Face-on view (estimated)", not a generic "side
    view" (see _scrubber_panel_titles). The estimated-angle panel's axis is MediaPipe's
    own estimated depth -- the weaker, noisier signal this project is about -- so its
    title says "estimated" rather than implying equal reliability."""
    x_label, y_label, z_label = _axis_labels(angle)
    n = len(frames)
    ts = [f.timestamp for f in frames]
    n_skeleton_traces = len(_SKELETON_GROUP_COLORS) + 1

    video_h, video_w = frames[0].frame.shape[:2]
    video_aspect = video_w / video_h
    panel_height = 480
    video_panel_width = round(panel_height * video_aspect)
    # +60 for the skeleton panels' y-axis title/ticks, which the video panel hides.
    skeleton_panel_width = video_panel_width + 60

    primary_title, secondary_title = _scrubber_panel_titles(angle)
    ranges = _global_axis_ranges(frames)

    fig_video = go.Figure(go.Image(source=_thumbnail_data_uri(frames[0])))
    fig_video.frames = [
        go.Frame(name=str(i), data=[go.Image(source=_thumbnail_data_uri(frames[i]))], traces=[0])
        for i in range(n)
    ]
    fig_video.update_xaxes(visible=False)
    fig_video.update_yaxes(visible=False, scaleanchor="x")
    fig_video.update_layout(
        title=_title("Video"), font=_FONT,
        width=video_panel_width, height=panel_height, margin=dict(t=40, b=10, l=10, r=10),
    )

    def skeleton_figure(x_attr: str, title: str, x_range: list, y_range: list) -> go.Figure:
        fig = go.Figure(_skeleton_traces_2d(frames[0].landmarks, x_attr))
        fig.frames = [
            go.Frame(name=str(i), data=_skeleton_traces_2d(frames[i].landmarks, x_attr), traces=list(range(n_skeleton_traces)))
            for i in range(n)
        ]
        # No 1:1 scaleanchor here: z's much wider, noisier spread would stretch the
        # estimated-angle panel's y-range and squash the body into a sliver.
        fig.update_xaxes(title_text=(x_label if x_attr == "x" else z_label), range=x_range, autorange=False)
        fig.update_yaxes(title_text=y_label, range=y_range, autorange=False)
        fig.update_layout(
            title=_title(title), font=_FONT,
            width=skeleton_panel_width, height=panel_height, margin=dict(t=40, b=40, l=55, r=10),
        )
        return fig

    fig_front = skeleton_figure("x", primary_title, ranges["xaxis"]["range"], ranges["yaxis"]["range"])
    fig_est = skeleton_figure("z", secondary_title, ranges["zaxis"]["range"], ranges["yaxis"]["range"])

    html_video = fig_video.to_html(full_html=False, include_plotlyjs=True, config=config, div_id="chart-scrubber-video", auto_play=False)
    html_front = fig_front.to_html(full_html=False, include_plotlyjs=False, config=config, div_id="chart-scrubber-front", auto_play=False)
    html_est = fig_est.to_html(full_html=False, include_plotlyjs=False, config=config, div_id="chart-scrubber-est", auto_play=False)

    template = _TEMPLATE_ENV.get_template("scrubber_controls.html")
    return template.render(
        html_video=html_video, html_front=html_front, html_est=html_est,
        max_frame=n - 1, timestamps_json=json.dumps([round(t, 3) for t in ts]),
        font_family=_FONT["family"], colors=_COLORS,
    )

def issues_grid_figure(phases: PhaseMap, metrics: list[Metrics], club: str, swing_type: str, angle: str) -> go.Figure:
    """Pass/warn/error heatmap with hover text showing the measured value and ideal
    range per cell."""
    ranges = IDEAL_RANGES[swing_type][club]
    address_m = metrics[phases["address"]]
    metric_keys = list(ranges.keys())
    phase_names = list(phases.keys())

    status = [[0] * len(phase_names) for _ in metric_keys]
    hover = [[""] * len(phase_names) for _ in metric_keys]
    for i, metric_key in enumerate(metric_keys):
        if angle == "dtl" and metric_key in _FACEON_METRICS: continue
        if angle == "face_on" and metric_key in _DTL_METRICS: continue
        for j, phase in enumerate(phase_names):
            ideal = ranges[metric_key].get(phase)
            if ideal is None:
                hover[i][j] = f"{metric_key} @ {phase}<br>n/a for this phase"
                continue
            value = _get(metrics[phases[phase]], metric_key, address_m)
            if value is None:
                hover[i][j] = f"{metric_key} @ {phase}<br>no data"
                continue
            lo, hi = ideal
            if lo <= value <= hi:
                status[i][j] = 1
            else:
                deviation = min(abs(value - lo), abs(value - hi))
                status[i][j] = 3 if deviation > (hi - lo) * 0.5 else 2
            hover[i][j] = f"{metric_key} @ {phase}<br>measured={value:.2f}<br>ideal=({lo:.2f}, {hi:.2f})"

    fig = go.Figure(go.Heatmap(
        z=status, x=phase_names, y=metric_keys, hoverinfo="text", text=hover,
        colorscale=[
            [0, _COLORS["grid_na"]], [0.33, _COLORS["grid_na"]],
            [0.33, _COLORS["grid_pass"]], [0.66, _COLORS["grid_pass"]],
            [0.66, _COLORS["grid_warn"]], [0.83, _COLORS["grid_warn"]],
            [0.83, _COLORS["grid_error"]], [1, _COLORS["grid_error"]],
        ],
        zmin=0, zmax=3, showscale=False, xgap=2, ygap=2,
    ))
    fig.update_layout(
        title=_title(f"Issue grid -- club={club} swing_type={swing_type} angle={angle}<br><sub>gray=n/a  green=pass  orange=warn  red=error</sub>"),
        font=_FONT, height=120 + 30 * len(metric_keys), margin=dict(t=80, b=40),
    )
    return fig
