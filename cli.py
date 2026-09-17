import argparse
from pathlib import Path

from config import CLUBS, LM_L_HIP, LM_L_SHOULDER, LM_R_HIP, LM_R_SHOULDER
from pose.detector import FramePose, detect
from pose.smoother import smooth
from swing.metrics import compute_all as compute_metrics
from swing.phases import PhaseMap, detect_phases
from swing.rotation import rotation_from_depth, rotation_from_span
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

    address_frame = frames[phases["address"]]
    address_m = metrics[phases["address"]]

    rows = []
    for phase, idx in phases.items():
        frame, m = frames[idx], metrics[idx]
        acos_hip = rotation_from_span(m.hip_span, address_m.hip_span)
        acos_sho = rotation_from_span(m.shoulder_span, address_m.shoulder_span)
        z_hip = rotation_from_depth(
            frame.landmarks[LM_L_HIP], frame.landmarks[LM_R_HIP],
            address_frame.landmarks[LM_L_HIP], address_frame.landmarks[LM_R_HIP],
        )
        z_sho = rotation_from_depth(
            frame.landmarks[LM_L_SHOULDER], frame.landmarks[LM_R_SHOULDER],
            address_frame.landmarks[LM_L_SHOULDER], address_frame.landmarks[LM_R_SHOULDER],
        )
        rows.append((phase, acos_hip, z_hip, acos_sho, z_sho))

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Golf swing posture tracker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--video", required=True, help="Path to swing video")
        sp.add_argument("--swing-type", default="full", choices=["full", "partial", "pitch", "chip"])

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


if __name__ == "__main__":
    main()
