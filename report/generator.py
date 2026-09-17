import base64
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from analysis.analyzer import Issue
from pose.detector import FramePose
from swing.metrics import Metrics
from swing.phases import PhaseMap
from swing.rotation import rotation_comparison_rows
from swing.state import SwingState
from visualization import visualization as viz

_TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

def _b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")

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
    """Assemble every chart the CLI can produce plus the analyzer's issues into one
    self-contained HTML report. Charts are base64-embedded; the annotated skeleton video
    is written alongside the report and linked, not embedded (too large for a data URI)."""
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        rotation_path = tmp_dir / "rotation_check.png"
        viz.plot_rotation_comparison(rotation_comparison_rows(frames, phases, metrics), str(rotation_path))

        birdseye_rotation_path = tmp_dir / "birdseye_rotation.png"
        birdseye_filmstrip_path = tmp_dir / "birdseye_filmstrip.png"
        viz.plot_birdseye_rotation(frames, phases, str(birdseye_rotation_path))
        viz.plot_birdseye_filmstrip(frames, phases, str(birdseye_filmstrip_path))

        metrics_chart_path = tmp_dir / "metrics_chart.png"
        viz.plot_metrics_timeseries(frames, metrics, phases, str(metrics_chart_path))

        state_timeseries_path = tmp_dir / "state_timeseries.png"
        state_trajectories_path = tmp_dir / "state_trajectories.png"
        viz.plot_state_timeseries(states, phases, str(state_timeseries_path))
        viz.plot_state_trajectories(states, str(state_trajectories_path))

        issues_grid_path = tmp_dir / "issues_grid.png"
        viz.plot_issues_grid(phases, metrics, club, swing_type, angle, str(issues_grid_path))

        charts = {
            "rotation_check": _b64_png(rotation_path),
            "birdseye_rotation": _b64_png(birdseye_rotation_path),
            "birdseye_filmstrip": _b64_png(birdseye_filmstrip_path),
            "metrics_chart": _b64_png(metrics_chart_path),
            "state_timeseries": _b64_png(state_timeseries_path),
            "state_trajectories": _b64_png(state_trajectories_path),
            "issues_grid": _b64_png(issues_grid_path),
        }

    video_filename = out_path.stem + "_annotated.mp4"
    fps = len(frames) / frames[-1].timestamp if frames[-1].timestamp else 30.0
    viz.export_annotated_video(frames, phases, str(out_path.parent / video_filename), fps)

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
        video_filename=video_filename,
    )
    out_path.write_text(html)
