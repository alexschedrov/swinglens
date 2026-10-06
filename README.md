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

### charts
Exports the blog post's charts (rotation check, bird's-eye view, state time series, state trajectories) as Plotly JSON files, one per chart, for rendering on a web page with `Plotly.newPlot`. Titles are dropped and backgrounds are transparent so the page supplies its own captions and theme.
```
uv run python cli.py charts --face-on samples/source/rory/rory_face_on_driver.mp4 --output samples/annotated/rory/charts
```
Same video-source rule as `export`.

### video downloads
```bash
yt-dlp -f "bv*[height<=720]+ba/b[height<=720]" --merge-output-format mp4 -o "downloaded.%(ext)s" "https://www.youtube.com/watch?v=P3YksJdejog"
```
To grab only part of the video, add `--download-sections` (accepts `HH:MM:SS` or raw seconds) plus `--force-keyframes-at-cuts` so the cut lands exactly on those timestamps instead of the nearest keyframe:
```bash
yt-dlp -f "bv*[height<=720]+ba/b[height<=720]" --merge-output-format mp4 --download-sections "*00:01:30-00:02:00" --force-keyframes-at-cuts -o "downloaded.%(ext)s" "https://www.youtube.com/watch?v=P3YksJdejog"
```

### video crop
Split a side-by-side split-screen video into its two camera angles (adjust the fractions if the seam isn't 50/50 — extract one frame first to check):
```bash
ffmpeg -i downloaded.mp4 -filter:v "crop=iw*0.6:ih:0:0" -c:a copy left.mp4
ffmpeg -i downloaded.mp4 -filter:v "crop=iw*0.4:ih:iw*0.6:0" -c:a copy right.mp4
```
To trim a time range while cropping, add `-ss START -t DURATION` before `-i` (fast seek to the nearest keyframe, then decode):
```bash
ffmpeg -ss 00:01:30 -i downloaded.mp4 -t 30 -filter:v "crop=iw*0.6:ih:0:0" -c:a copy left.mp4
```