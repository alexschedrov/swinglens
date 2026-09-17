import argparse
from pathlib import Path

from analysis.analyzer import analyze
from config import CLUBS
from pose.detector import FramePose, detect
from pose.smoother import smooth
from report.export import export_states
from report.generator import generate_report
from swing.metrics import compute_all as compute_metrics
from swing.phases import PhaseMap, detect_phases
from swing.rotation import rotation_comparison_rows
from swing.state import compute_all as compute_states
from visualization import visualization as viz

def _run_pipeline(video: str, swing_type: str) -> tuple[list[FramePose], PhaseMap]:
    print(f"[1/3] detecting pose: {video}")
    frames = detect(video)

    print(f"[2/3] smoothing landmarks ({len(frames)} frames)")
    frames = smooth(frames)

    print("[3/3] detecting swing phases")
    phases = detect_phases(frames, swing_type=swing_type)
    for phase, idx in phases.items():
        print(f"  {phase:<16} frame {frames[idx].frame_idx:>4}  t={frames[idx].timestamp:.2f}s")

    return frames, phases

def run_annotate(video: str, swing_type: str, output: str) -> None:
    frames, phases = _run_pipeline(video, swing_type)
    fps = len(frames) / frames[-1].timestamp if frames[-1].timestamp else 30.0
    viz.export_annotated_video(frames, phases, output, fps)
    print(f"saved annotated video -> {output}")

def run_rotation_check(video: str, swing_type: str, output: str) -> None:
    """Compare the acos-trick rotation (from shrinking projected span) against a
    z-derived rotation (from MediaPipe's own depth coordinate) at every phase.
    A self-consistency check between two signals from the same pose model, not a
    validation against ground truth."""
    frames, phases = _run_pipeline(video, swing_type)
    metrics = compute_metrics(frames)
    rows = rotation_comparison_rows(frames, phases, metrics)

    print(f"\n{'phase':<16}{'hip (acos)':>12}{'hip (z)':>10}{'sho (acos)':>12}{'sho (z)':>10}")
    for phase, acos_hip, z_hip, acos_sho, z_sho in rows:
        print(f"{phase:<16}{acos_hip:>11.1f}{z_hip:>10.1f}{acos_sho:>12.1f}{z_sho:>10.1f}")

    viz.plot_rotation_comparison(rows, output)
    print(f"saved comparison chart -> {output}")

def run_birdseye(video: str, swing_type: str, output_dir: str) -> None:
    frames, phases = _run_pipeline(video, swing_type)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rotation_path = out_dir / "birdseye_rotation.png"
    filmstrip_path = out_dir / "birdseye_filmstrip.png"
    viz.plot_birdseye_rotation(frames, phases, str(rotation_path))
    viz.plot_birdseye_filmstrip(frames, phases, str(filmstrip_path))
    print(f"saved -> {rotation_path}")
    print(f"saved -> {filmstrip_path}")

def run_metrics_chart(video: str, swing_type: str, output: str) -> None:
    frames, phases = _run_pipeline(video, swing_type)
    metrics = compute_metrics(frames)
    viz.plot_metrics_timeseries(frames, metrics, phases, output)
    print(f"saved metrics chart -> {output}")

def run_state_chart(video: str, swing_type: str, output_dir: str) -> None:
    frames, phases = _run_pipeline(video, swing_type)
    metrics = compute_metrics(frames)
    states = compute_states(frames, metrics, phases)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = out_dir / "state_timeseries.png"
    trajectories_path = out_dir / "state_trajectories.png"
    viz.plot_state_timeseries(states, phases, str(timeseries_path))
    viz.plot_state_trajectories(states, str(trajectories_path))
    print(f"saved -> {timeseries_path}")
    print(f"saved -> {trajectories_path}")

def run_issues_chart(video: str, swing_type: str, club: str, angle: str, output: str) -> None:
    frames, phases = _run_pipeline(video, swing_type)
    metrics = compute_metrics(frames)
    viz.plot_issues_grid(phases, metrics, club, swing_type, angle, output)
    print(f"saved issues grid -> {output}")

def run_export(video: str, swing_type: str, club: str, angle: str, output: str) -> None:
    frames, phases = _run_pipeline(video, swing_type)
    metrics = compute_metrics(frames)
    states = compute_states(frames, metrics, phases)

    issues = analyze(phases, metrics, club, swing_type, angle)
    if issues:
        print(f"\n{len(issues)} issue(s) found:")
        for issue in issues:
            print(f"  [{issue.severity:<5}] {issue.phase:<16}{issue.metric:<24}"
                  f"measured={issue.measured:.2f}  ideal=({issue.ideal_min:.2f}, {issue.ideal_max:.2f})")
    else:
        print("\nno issues found")

    export_states(states, output)
    print(f"saved state trajectory ({len(states)} frames) -> {output}")

def run_report(video: str, swing_type: str, club: str, angle: str, output: str) -> None:
    frames, phases = _run_pipeline(video, swing_type)
    metrics = compute_metrics(frames)
    states = compute_states(frames, metrics, phases)
    issues = analyze(phases, metrics, club, swing_type, angle)

    generate_report(frames, phases, metrics, states, issues, video, club, swing_type, angle, output)
    print(f"saved report -> {output}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Golf swing posture tracker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--video", required=True, help="Path to swing video")
        sp.add_argument("--swing-type", default="full", choices=["full", "partial", "pitch", "chip"])

    def add_angle_aware_video_source(sp: argparse.ArgumentParser) -> None:
        video_source = sp.add_mutually_exclusive_group(required=True)
        video_source.add_argument("--video", help="Path to swing video (camera angle unspecified)")
        video_source.add_argument("--dtl", help="Path to a down-the-line angle video")
        video_source.add_argument("--face-on", dest="face_on", help="Path to a face-on angle video")
        sp.add_argument("--swing-type", default="full", choices=["full", "partial", "pitch", "chip"])

    def resolve_angle_aware_video(args: argparse.Namespace) -> tuple[str, str]:
        video = args.dtl or args.face_on or args.video
        angle = "dtl" if args.dtl else "face_on" if args.face_on else "unknown"
        return video, angle

    annotate = subparsers.add_parser("annotate", help="Export a skeleton-overlay video with phase markers")
    add_common(annotate)
    annotate.add_argument("--output", help="Path for the annotated output video")

    rotation_check = subparsers.add_parser("rotation-check", help="Compare acos-trick vs z-derived rotation")
    add_common(rotation_check)
    rotation_check.add_argument("--output", default="rotation_check.png", help="Path for the comparison chart")

    birdseye = subparsers.add_parser("birdseye", help="Bird's-eye x/z rotation visualizations")
    add_common(birdseye)
    birdseye.add_argument("--output-dir", default="birdseye_output")

    metrics_chart = subparsers.add_parser("metrics-chart", help="Plot Metrics over time")
    add_common(metrics_chart)
    metrics_chart.add_argument("--output", default="metrics_chart.png")

    state_chart = subparsers.add_parser("state-chart", help="Plot SwingState over time and as trajectories")
    add_common(state_chart)
    state_chart.add_argument("--output-dir", default="state_output")

    issues_chart = subparsers.add_parser("issues-chart", help="Visualize analyzer issues as a pass/warn/error grid")
    add_common(issues_chart)
    issues_chart.add_argument("--club", default="iron", choices=CLUBS)
    issues_chart.add_argument("--angle", default="unknown", choices=["dtl", "face_on", "unknown"])
    issues_chart.add_argument("--output", default="issues_grid.png")

    export = subparsers.add_parser("export", help="Run the full pipeline and export the per-frame state trajectory")
    add_angle_aware_video_source(export)
    export.add_argument("--club", default="iron", choices=CLUBS)
    export.add_argument("--output", required=True, help="Output path for the state trajectory (.json or .csv)")

    report = subparsers.add_parser("report", help="Run the full pipeline and generate a self-contained HTML report")
    add_angle_aware_video_source(report)
    report.add_argument("--club", default="iron", choices=CLUBS)
    report.add_argument("--output", default="report.html", help="Output path for the HTML report")

    args = parser.parse_args()

    if args.command == "annotate":
        output = args.output or str(Path(args.video).with_stem(Path(args.video).stem + "_annotated"))
        run_annotate(args.video, args.swing_type, output)
    elif args.command == "rotation-check":
        run_rotation_check(args.video, args.swing_type, args.output)
    elif args.command == "birdseye":
        run_birdseye(args.video, args.swing_type, args.output_dir)
    elif args.command == "metrics-chart":
        run_metrics_chart(args.video, args.swing_type, args.output)
    elif args.command == "state-chart":
        run_state_chart(args.video, args.swing_type, args.output_dir)
    elif args.command == "issues-chart":
        run_issues_chart(args.video, args.swing_type, args.club, args.angle, args.output)
    elif args.command == "export":
        video, angle = resolve_angle_aware_video(args)
        run_export(video, args.swing_type, args.club, angle, args.output)
    elif args.command == "report":
        video, angle = resolve_angle_aware_video(args)
        run_report(video, args.swing_type, args.club, angle, args.output)


if __name__ == "__main__":
    main()
