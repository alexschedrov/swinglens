## Commands

All commands run via `uv run python cli.py <command> [flags]`. `--video` and `--swing-type` (`full`/`partial`/`pitch`/`chip`, default `full`) are shared by every command.

### annotate
Skeleton-overlay video with a phase label burned into every frame.
```
uv run python cli.py annotate --video samples/source/face_on.mp4 --output samples/annotated/face_on_annotated.mp4
```
`--output` defaults to `<video>_annotated.<ext>` next to the input if omitted.

### rotation-check
Compares the acos-trick rotation (shrinking projected span) against a z-derived rotation (MediaPipe's own depth) at every phase — a self-consistency check, not a ground-truth validation.
```
uv run python cli.py rotation-check --video samples/source/face_on.mp4 --output samples/annotated/rotation_check.png
```

### birdseye
Bird's-eye (x/z plane) visualizations of the hip/shoulder line rotating through the swing: a fan diagram and a camera-frame/bird's-eye filmstrip, one column per phase.
```
uv run python cli.py birdseye --video samples/source/face_on.mp4 --output-dir samples/annotated/birdseye
```
Writes `birdseye_rotation.png` and `birdseye_filmstrip.png` into `--output-dir`.

### metrics-chart
Plots every `Metrics` field (spine angle, knee flex, hip/shoulder span, head position, hip sway) over time, with phase boundaries marked.
```
uv run python cli.py metrics-chart --video samples/source/face_on.mp4 --output samples/annotated/metrics_chart.png
```

### state-chart
Plots `SwingState` rotation/angle fields and angular velocities over time, plus 2D lead-wrist and COM-proxy trajectories.
```
uv run python cli.py state-chart --video samples/source/face_on.mp4 --output-dir samples/annotated/state
```
Writes `state_timeseries.png` and `state_trajectories.png` into `--output-dir`.

### issues-chart
Pass/warn/error grid: every (metric, phase) cell `IDEAL_RANGES` defines, colored by whether the measured value is within range for the given club/swing-type/angle.
```
uv run python cli.py issues-chart --video samples/source/dtl.mp4 --club iron --angle dtl --output samples/annotated/issues_grid.png
```
`--club` choices: `driver`, `wood`, `hybrid`, `iron`, `short_iron`, `wedge`. `--angle` choices: `dtl`, `face_on`, `unknown`.

### export
Runs the full pipeline (pose detection → smoothing → phase detection → metrics → analysis) and exports the per-frame `SwingState` trajectory to JSON or CSV, chosen by `--output`'s extension. Detected issues (measured values outside `IDEAL_RANGES` for the given club/swing-type/angle) are printed to the console.
```
uv run python cli.py export --video samples/source/dtl.mp4 --club iron --output samples/annotated/state_trajectory.json
```
Takes exactly one video source — `--video`, `--dtl`, or `--face-on` — the latter two also set the camera angle used by the analyzer. With `--video` alone, the angle is `unknown` and the analyzer evaluates every metric (both DTL- and face-on-only), which can produce spurious issues for metrics that aren't geometrically valid from the actual camera angle — prefer `--dtl`/`--face-on` when you know it.