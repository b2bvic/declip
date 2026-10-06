# declip

You can use `declip` to plan filler, silence, and retake cuts in local audio or video.
You review the transcript and proposed cuts on a local browser page.
You then render media or export files for your video editor.
Your source stays unchanged.

[Project page](https://scalewithsearch.com/code/declip)

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.11 or later, and `ffmpeg` with `ffprobe`.
Use the Git URL for version 0.5.0.
You cannot install this version from PyPI.

On macOS 14 or later with Apple silicon:

```bash
brew install ffmpeg
uv tool install "declip[mac] @ git+https://github.com/b2bvic/declip@v0.5.0"
```

On Linux with NVIDIA hardware (assumed, not verified):

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
uv tool install "declip[cuda] @ git+https://github.com/b2bvic/declip@v0.5.0"
```

For CPU transcription on macOS or Linux:

```bash
uv tool install "declip[cpu] @ git+https://github.com/b2bvic/declip@v0.5.0"
```

Run `uv tool update-shell` if your shell cannot find `declip`.
Run `declip doctor` to check tools and backends.
For a machine used only for review, render, or export, install the core package without an extra:

```bash
uv tool install "declip @ git+https://github.com/b2bvic/declip@v0.5.0"
```

Windows is planned for a later release and is untested.
See [platform setup](docs/platforms.md) for CUDA and remote review.

## Set up your rig

Create a named profile with a sample from your recording equipment:

```bash
declip setup --name desk --sample sample.mp4
declip rig show desk
```

Use a sample of at least ten seconds.
Answer questions about your microphone, room, loudness, and output destination.
Setup measures the sample and detects your hardware.
Your equipment tier controls processing, bitrate, and output files.
Use `--rig desk` to select that profile for a run.
The first profile becomes your default.
Use `--make-default` to change the default later.

For setup without prompts:

```bash
declip setup --name desk --sample sample.mp4 --mic usb --room normal --destination publish --yes
```

`--yes` accepts setup defaults only.
It cannot bypass review.
See [rig profiles](docs/rig-profiles.md) for tiers, flags, and precedence.

## Plan, review, and render

```bash
declip plan clip.mp4 --rig desk
declip review clip.mp4
declip render clip.mp4
```

Planning writes `clip.declip.json` with proposed cuts.
Review is required for every run and every tier, including plans with no cuts.
Accept or reject every proposal, then select **Finish review**.
Render and export refuse pending or stale review with exit code 3.
A changed source also requires a new plan and review.
No flag, profile, or config setting bypasses the gate.

On the review page, click a word to seek.
Click a cut to accept or reject it.
Select a word range and press `c` to add a manual cut.
Use `j` and `k` to move between cuts, `a` to accept, and `x` to reject.
Use Space to play or pause.
Press `p` to preview playback around the current cut with accepted intervals skipped.
Your changes save in the same edit list.

Render writes `clip_clean.mp4`, or `clip_clean.mov` for a MOV source.
It matches the source codec, dimensions, frame rate, and color tags by default.
A 10-bit source stays 10-bit unless you pass `--allow-8bit` with a compatible codec choice.
Audio passes through trim, enhancement, EQ, two-pass loudness normalization, and one encode.
Render keeps the source sample rate and channel count.
It cuts additional audio tracks without processing their sound.
It copies global metadata and drops subtitle, data, attachment, and timecode streams.
Render can conform variable-frame-rate video internally.

Without a rig or preset choice, you get `raw`, which applies loudness normalization only.
Choose `voice`, `natural`, `podcast`, or `none` with `--preset`.
`none` disables EQ and loudness.
See the [edit-list reference](docs/edit-list.md) for reruns, source binding, and migration.

## Export for your editor

After review:

```bash
declip export clip.mp4 --format edl
declip export clip.mp4 --format srt
declip export clip.mp4 --format markers
```

Exports use `clip.declip.edl`, `.fcpxml`, `.srt`, and `.markers.json`.
SRT captions contain retained words on the output timeline.
Exports never overwrite `clip.declip.json`.
Existing outputs require `--overwrite`.

EDL is the pro-tier default for NLE output.
Choose `--destination nle` during setup to export for your editor.
Basic and creator NLE profiles also produce a processed PCM 24-bit sidecar WAV.
Pro NLE profiles preserve source media without audio processing or a sidecar.
FCPXML remains an explicit choice with `--format fcpxml`.
FCPXML passed structural tests, but import into an editor is unverified.

EDL and FCPXML require constant-frame-rate video at a standard rate.
If export refuses variable-frame-rate video, follow its conform command.
Plan and review the conformed source again.

## Use older commands

```bash
declip process clip.mp4
declip process clip.mp4 --execute
declip clean clip.mp4 --execute
declip --execute process clip.mp4
```

`process` without `--execute` reports proposed cuts and writes nothing.
`process --execute` and `clean --execute` write an edit list and stop for review.
The older `--export` option also stops at the edit list.
Run `review`, then `render` or `export` yourself.
You can place compatibility options before or after the subcommand.
Subcommand values take precedence.

Whole-file `enhance --execute` processes audio without cuts.
It uses a separate path that cannot change the timeline.
Use `--device cpu` for CPU transcription; the older `--cpu` flag has no effect.

## Check platform support

The following states describe local evidence as of 2026.10.06.
A verified test does not certify every device or input file.

| Function | macOS arm64 | Linux with NVIDIA | CPU on macOS or Linux |
|---|---|---|---|
| Transcription | Verified MLX on Metal: 18/18 fillers, zero false hits | Assumed, not verified | Verified faster int8 on macOS: 16/18 fillers, one false hit; Linux assumed |
| 8-bit render | Verified H.264 through VideoToolbox | Assumed NVENC path | Verified software H.264 on macOS; Linux assumed |
| 10-bit render | Verified VideoToolbox HEVC Main 10, `yuv420p10le` | Assumed NVENC path | Verified software HEVC Main 10 on macOS, `yuv420p10le`; Linux assumed |
| A/V sync over 250 keeps | Verified VideoToolbox: maximum offset 0.498 frames | Assumed, not verified | Verified software on macOS: maximum offset 0.498 frames; Linux assumed |
| Review and render workflow | Verified local generated-media browser test | Assumed | Verified locally on macOS; Linux assumed |
| Hardware receipt | [Metal receipt](docs/receipts/macos-metal.json), passed | Waived for 0.5.0 while hardware is unreachable; remains assumed | [CPU receipt](docs/receipts/macos-cpu.json), passed on macOS only |
| Hosted CI | Assumed; `macos-latest` job configured | Assumed; `ubuntu-latest` job configured (no GPU) | Both runner jobs configured; hosted results assumed |

Windows is planned for a later release and is untested.
Linux with NVIDIA is assumed, not verified; no Linux hardware receipt exists.

## Understand the limits

Filler and retake detection use heuristics.
A historical retake check found one of nine known retakes.
Use manual cuts for missed retakes.
`--min-confidence` marks low-confidence proposals; it never decides a cut for you.
Only English filler rules ship with this version.
Other languages can use gaps and retakes, or your own filler file.

The filler-prompt probe was inconclusive on a synthetic clip for both backends.
No real-speech receipt exists.
Prompt persistence after thirty seconds remains unproven.
The default `auto` prompt mode uses `initial`.
See [models and prompt behavior](docs/models.md) before interpreting a transcript.

Caption burn-in, LUT processing, forced 4K output, and ProRes output are outside this version.
See [security and audit results](SECURITY.md) before using shared edit lists or changing dependencies.

## Develop

```bash
uv sync --locked --extra dev
uv run --locked --extra dev pytest -q --require ffmpeg --require libx265
uv run --locked --extra dev ruff check src tests
```

CI targets `ubuntu-latest` and `macos-latest` with Python 3.11.
Tests use synthetic transcripts and generated media without model downloads.
See [contribution instructions](CONTRIBUTING.md) for isolation and audit commands.

## Related repositories

- [owned-record](https://github.com/b2bvic/owned-record): agent context and session history in files you control, including `web2md`.
- [ops-scripts](https://github.com/b2bvic/ops-scripts): operator scripts, including the X bookmark import.
- [seo-checks](https://github.com/b2bvic/seo-checks): SEO page checks from one command.

## License

MIT. See [LICENSE](LICENSE).
