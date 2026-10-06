import argparse
from pathlib import Path

from analysis.analyzer import analyze
from config import CLUBS
from pose.detector import FramePose, detect
from pose.smoother import smooth
from report.export import export_states
from report.generator import export_charts, generate_report
from swing.metrics import compute_all as compute_metrics
from swing.phases import PhaseMap, detect_phases
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

def run_charts(video: str, swing_type: str, angle: str, output: str) -> None:
    frames, phases = _run_pipeline(video, swing_type)
    metrics = compute_metrics(frames)
    states = compute_states(frames, metrics, phases)
    for path in export_charts(frames, phases, metrics, states, angle, output):
        print(f"saved chart -> {path}")

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

    export = subparsers.add_parser("export", help="Run the full pipeline and export the per-frame state trajectory")
    add_angle_aware_video_source(export)
    export.add_argument("--club", default="iron", choices=CLUBS)
    export.add_argument("--output", required=True, help="Output path for the state trajectory (.json or .csv)")

    report = subparsers.add_parser("report", help="Run the full pipeline and generate a self-contained HTML report")
    add_angle_aware_video_source(report)
    report.add_argument("--club", default="iron", choices=CLUBS)
    report.add_argument("--output", default="report.html", help="Output path for the HTML report")

    charts = subparsers.add_parser("charts", help="Run the full pipeline and export charts as Plotly JSON")
    add_angle_aware_video_source(charts)
    charts.add_argument("--output", required=True, help="Output directory for the chart JSON files")

    args = parser.parse_args()

    if args.command == "annotate":
        output = args.output or str(Path(args.video).with_stem(Path(args.video).stem + "_annotated"))
        run_annotate(args.video, args.swing_type, output)
    elif args.command == "export":
        video, angle = resolve_angle_aware_video(args)
        run_export(video, args.swing_type, args.club, angle, args.output)
    elif args.command == "report":
        video, angle = resolve_angle_aware_video(args)
        run_report(video, args.swing_type, args.club, angle, args.output)
    elif args.command == "charts":
        video, angle = resolve_angle_aware_video(args)
        run_charts(video, args.swing_type, angle, args.output)


if __name__ == "__main__":
    main()
