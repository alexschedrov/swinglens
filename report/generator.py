from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from analysis.analyzer import Issue
from pose.detector import FramePose
from report.charts import (
    birdseye_rotation_figure,
    issues_grid_figure,
    metrics_timeseries_figure,
    rotation_comparison_figure,
    state_timeseries_figure,
    state_trajectories_figure,
    video_scrubber_html,
)
from swing.metrics import Metrics
from swing.phases import PhaseMap
from swing.rotation import rotation_comparison_rows
from swing.state import SwingState

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"
_CHART_CONFIG = {"displaylogo": False, "responsive": True}
# The scrubber uses fixed pixel dimensions matched to the video's own aspect ratio
# (see video_scrubber_html) -- "responsive" resizing would stretch it to the
# container's width while keeping its height fixed, breaking that aspect match.
_SCRUBBER_CONFIG = {"displaylogo": False, "responsive": False}

def generate_report(
    frames: list[FramePose],
    phases: PhaseMap,
    metrics: list[Metrics],
    states: list[SwingState],
    issues: list[Issue],
    video: str,
    club: str,
    swing_type: str,
    angle: str,
    output: str,
) -> None:
    """Assemble every interactive chart the CLI can produce plus the analyzer's issues
    into one self-contained local HTML file. Each chart bundles plotly.js inline (no CDN,
    no separate files) so the report is a single file that works fully offline."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # plotly.js (~4.8MB) is bundled only once, in the first chart rendered (the
    # scrubber) -- later charts' inline scripts reuse that global Plotly object.
    def chart_html(fig, div_id: str, include_js: bool, config: dict = _CHART_CONFIG) -> str:
        # auto_play=False: plotly.py's default auto-plays figures with animation frames on load.
        return fig.to_html(full_html=False, include_plotlyjs=include_js, config=config, div_id=div_id, auto_play=False)

    charts = {
        "scrubber": video_scrubber_html(frames, angle, _SCRUBBER_CONFIG, video),
        "rotation": chart_html(rotation_comparison_figure(rotation_comparison_rows(frames, phases, metrics)), "chart-rotation", False),
        "birdseye": chart_html(birdseye_rotation_figure(frames, phases, angle), "chart-birdseye", False),
        "metrics": chart_html(metrics_timeseries_figure(frames, metrics, phases), "chart-metrics", False),
        "state_ts": chart_html(state_timeseries_figure(states, phases), "chart-state-ts", False),
        "state_traj": chart_html(state_trajectories_figure(states, angle), "chart-state-traj", False),
        "issues": chart_html(issues_grid_figure(phases, metrics, club, swing_type, angle), "chart-issues", False),
    }

    phase_rows = [
        {"phase": phase, "frame": frames[idx].frame_idx, "time": frames[idx].timestamp}
        for phase, idx in phases.items()
    ]

    env = Environment(loader=FileSystemLoader(_TEMPLATE_DIR), autoescape=select_autoescape())
    template = env.get_template("report.html")
    html = template.render(
        video=video,
        club=club,
        swing_type=swing_type,
        angle=angle,
        phase_rows=phase_rows,
        issues=issues,
        charts=charts,
    )
    out_path.write_text(html)

def export_charts(
    frames: list[FramePose],
    phases: PhaseMap,
    metrics: list[Metrics],
    states: list[SwingState],
    angle: str,
    out_dir: str,
) -> list[Path]:
    """Write the charts as Plotly JSON files, one per chart, for a page to
    render client-side with Plotly.newPlot. Titles are dropped and the background made
    transparent: the embedding page supplies its own captions and theme."""
    figures = {
        "rotation": rotation_comparison_figure(rotation_comparison_rows(frames, phases, metrics)),
        "birdseye": birdseye_rotation_figure(frames, phases, angle),
        "state_timeseries": state_timeseries_figure(states, phases),
        "state_trajectories": state_trajectories_figure(states, angle),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        fig.update_layout(title=None, paper_bgcolor="rgba(0,0,0,0)", margin_t=40)
        path = out / f"{name}.json"
        fig.write_json(path)
        paths.append(path)
    return paths
