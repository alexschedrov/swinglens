import base64
import json
import mimetypes
from pathlib import Path

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
from visualization.visualization import _SEGMENTS

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

# Usable width of the report's scrubber section (templates/report.html caps the page at
# 1600px, minus page and section padding); the scrubber's three panels must fit in it.
_SCRUBBER_ROW_WIDTH = 1440
_SCRUBBER_GAP = 16

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

def _skeleton_traces_2d(landmarks: list[Landmark], x_attr: str, flip_x: bool = False) -> list[go.Scatter]:
    """One 2D trace per bone-color group plus one joint-marker trace, in a fixed order so
    trace indices line up across every animation frame. x_attr picks which landmark
    coordinate is the plotted x-axis -- 'x' for a front view matching the camera, 'z' for
    a side view showing MediaPipe's estimated depth; y is always vertical (flipped so
    "up" is positive). flip_x negates the plotted x-axis -- a 180-degree turn used for the
    down-the-line-estimated panel so it reads as viewed from the golfer's left side,
    rather than whatever side MediaPipe's arbitrary z-sign convention happens to produce."""
    sign = -1 if flip_x else 1
    bone_traces = []
    for color, pairs in _SKELETON_GROUPS.items():
        xs: list = []
        ys: list = []
        for a_idx, b_idx in pairs:
            a, b = landmarks[a_idx], landmarks[b_idx]
            if a.visibility <= 0.4 or b.visibility <= 0.4:
                continue
            xs += [sign * getattr(a, x_attr), sign * getattr(b, x_attr), None]
            ys += [-a.y, -b.y, None] # image y grows downward; flip so "up" is positive
        bone_traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=3), showlegend=False))

    visible = [lm for lm in landmarks if lm.visibility > 0.4]
    joints_trace = go.Scatter(
        x=[sign * getattr(lm, x_attr) for lm in visible], y=[-lm.y for lm in visible],
        mode="markers", marker=dict(size=4, color=_COLORS["joint"]), showlegend=False,
    )
    return [*bone_traces, joints_trace]


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

def _skeleton_overlay_data(frames: list[FramePose]) -> tuple[list, list]:
    """Per-frame [x, y, visibility] landmarks plus [a, b, rgb] segments, for the video
    panel's canvas overlay to draw the same skeleton as visualization.draw_skeleton."""
    landmarks = [[[round(lm.x, 4), round(lm.y, 4), round(lm.visibility, 2)] for lm in fp.landmarks] for fp in frames]
    segments = [[a, b, f"rgb({c[2]},{c[1]},{c[0]})"] for (a, b), c in _SEGMENTS.items()]
    return landmarks, segments

def video_scrubber_html(frames: list[FramePose], angle: str, config: dict, video_path: str) -> str:
    """Synced three-panel scrubber, returned as ready-to-embed HTML: the source video with
    a skeleton canvas overlay, a flat view of the skeleton matching the video's own camera
    angle (x/y), and a flat view estimating the *other* camera angle (z/y).

    The video panel is a native <video> element playing the original file (embedded as
    base64), not per-frame images: the browser decodes it at full quality, and the file
    stays a fraction of the size of storing every frame as a JPEG. The skeleton is drawn
    on a <canvas> on top of it from the landmark JSON.

    One shared range input + play/pause button drives everything (see
    templates/scrubber_controls.html): the video's own playback clock picks the current
    frame, and the two skeleton panels follow via Plotly.animate.

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
    # Shrink the panels when needed so video + two skeleton panels (each +60px for the
    # y-axis title/ticks) + two gaps fit one row of _SCRUBBER_ROW_WIDTH.
    # The video sits under a 40px title with a 10px bottom margin, matching the skeleton
    # panels' title band.
    fixed_width = 2 * 60 + 2 * _SCRUBBER_GAP
    panel_height = min(480, round((_SCRUBBER_ROW_WIDTH - fixed_width) / (3 * video_aspect)) + 50)
    video_display_height = panel_height - 50
    video_panel_width = round(video_display_height * video_aspect)
    skeleton_panel_width = video_panel_width + 60

    primary_title, secondary_title = _scrubber_panel_titles(angle)
    ranges = _global_axis_ranges(frames)
    landmarks, segments = _skeleton_overlay_data(frames)
    # Seek to the middle of a frame, not its start, so the browser can't land on the previous one.
    seek_offset = float(np.median(np.diff(ts))) / 2 if n > 1 else 0.0

    def skeleton_figure(x_attr: str, title: str, x_range: list, y_range: list, flip_x: bool = False) -> go.Figure:
        fig = go.Figure(_skeleton_traces_2d(frames[0].landmarks, x_attr, flip_x))
        fig.frames = [
            go.Frame(name=str(i), data=_skeleton_traces_2d(frames[i].landmarks, x_attr, flip_x), traces=list(range(n_skeleton_traces)))
            for i in range(n)
        ]
        plotted_range = [-x_range[1], -x_range[0]] if flip_x else x_range
        # No 1:1 scaleanchor here: z's much wider, noisier spread would stretch the
        # estimated-angle panel's y-range and squash the body into a sliver.
        fig.update_xaxes(title_text=(x_label if x_attr == "x" else z_label), range=plotted_range, autorange=False)
        fig.update_yaxes(title_text=y_label, range=y_range, autorange=False)
        fig.update_layout(
            title=_title(title), font=_FONT,
            width=skeleton_panel_width, height=panel_height, margin=dict(t=40, b=40, l=55, r=10),
        )
        return fig

    # When the estimated panel is standing in for a down-the-line view (i.e. the actual
    # footage is face-on), flip it 180 degrees so it reads as viewed from the golfer's
    # left side, rather than whatever side MediaPipe's arbitrary z-sign gives by default.
    fig_front = skeleton_figure("x", primary_title, ranges["xaxis"]["range"], ranges["yaxis"]["range"])
    fig_est = skeleton_figure("z", secondary_title, ranges["zaxis"]["range"], ranges["yaxis"]["range"], flip_x=(angle == "face_on"))

    html_front = fig_front.to_html(full_html=False, include_plotlyjs=True, config=config, div_id="chart-scrubber-front", auto_play=False)
    html_est = fig_est.to_html(full_html=False, include_plotlyjs=False, config=config, div_id="chart-scrubber-est", auto_play=False)

    template = _TEMPLATE_ENV.get_template("scrubber_controls.html")
    return template.render(
        html_front=html_front, html_est=html_est,
        video_b64=base64.b64encode(Path(video_path).read_bytes()).decode("ascii"),
        video_mime=mimetypes.guess_type(video_path)[0] or "video/mp4",
        video_width=video_panel_width, video_height=video_display_height, gap=_SCRUBBER_GAP,
        landmarks_json=json.dumps(landmarks, separators=(",", ":")),
        segments_json=json.dumps(segments), seek_offset=seek_offset,
        max_frame=n - 1, timestamps_json=json.dumps([round(t, 3) for t in ts]),
        font_family=_FONT["family"], colors=_COLORS, title_font_size=_TITLE_FONT["size"],
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
