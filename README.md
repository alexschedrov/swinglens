## Commands

All commands run via `uv run python cli.py <command> [flags]`. `--video` and `--swing-type` (`full`/`partial`/`pitch`/`chip`, default `full`) are shared by every command.

### annotate
Skeleton-overlay video with a phase label burned into every frame.
```
uv run python cli.py annotate --video samples/source/face_on.mp4 --output samples/annotated/face_on_annotated.mp4
```
`--output` defaults to `<video>_annotated.<ext>` next to the input if omitted.

### export
Runs the full pipeline (pose detection → smoothing → phase detection → metrics → analysis) and exports the per-frame `SwingState` trajectory to JSON or CSV, chosen by `--output`'s extension. Detected issues (measured values outside `IDEAL_RANGES` for the given club/swing-type/angle) are printed to the console.
```
uv run python cli.py export --video samples/source/dtl.mp4 --club iron --output samples/annotated/state_trajectory.json
```
Takes exactly one video source — `--video`, `--dtl`, or `--face-on` — the latter two also set the camera angle used by the analyzer. With `--video` alone, the angle is `unknown` and the analyzer evaluates every metric (both DTL- and face-on-only), which can produce spurious issues for metrics that aren't geometrically valid from the actual camera angle — prefer `--dtl`/`--face-on` when you know it.

### report
Runs the full pipeline once and generates a single self-contained, interactive HTML report (Plotly charts, no separate files): a synced video/skeleton scrubber, phase table, issues table + issues grid, the rotation self-consistency chart, bird's-eye view, the metrics chart, and the state charts.
```
uv run python cli.py report --dtl samples/source/dtl.mp4 --club iron --output samples/annotated/report/report.html
```
Same video-source rule as `export` (`--video`/`--dtl`/`--face-on`, mutually exclusive). `--output` defaults to `report.html`.
