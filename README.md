# declip

Local filler removal for talking-head video. No cloud, no subscription, no upload. Runs on Apple Silicon.

Replaces Descript's filler word removal + Studio Sound with a single Python script using Whisper (MLX) and ffmpeg.

## What it does

- Detects and removes filler words (um, uh, like, you know, basically, etc.)
- Compresses dead air gaps to a configurable maximum
- Detects retakes (repeated lines) and keeps the latest take
- Applies voice EQ presets matched to your mic profile
- Optional neural speech enhancement via Resemble Enhance
- Frame-accurate cuts with 15ms audio fades at every boundary
- Dry-run mode shows exactly what will be cut before processing

## Install

Requirements: macOS with Apple Silicon, Python 3.11+, ffmpeg with VideoToolbox.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Download declip
curl -o ~/.local/bin/declip https://raw.githubusercontent.com/b2bvic/declip/main/declip
chmod +x ~/.local/bin/declip
```

That's it. Dependencies (mlx-whisper, click) are resolved automatically by `uv` on first run.

## Usage

```bash
# Dry-run: see what would be cut (default)
declip process video.mp4

# Execute: process the video
declip process video.mp4 --execute

# Compress gaps longer than 500ms
declip process video.mp4 --max-gap 500 --execute

# Cut a specific range (e.g. off-topic conversation)
declip process video.mp4 --cut-range "570.5-695.9" --execute

# Use a different EQ preset
declip process video.mp4 --preset podcast --execute

# Just detect fillers (report only)
declip detect video.mp4

# Just transcribe
declip transcribe video.mp4
```

## EQ Presets

Presets live in `~/.config/declip/presets.json`. Ships with:

| Preset | Description |
|--------|-------------|
| `victor` | DJI Mic lav matched to Descript Studio Sound. Bass restore, presence lift, tight dynamics. |
| `natural` | Transparent. EQ only, no compression. |
| `podcast` | Broadcast standard, -16 LUFS, moderate compression. |
| `raw` | Loudness normalization only. |
| `none` | Pass-through. |

Create your own by adding entries to the JSON file.

## Options

```
--execute            Actually process (default is dry-run)
--preset, -p         EQ preset (default: victor)
--max-gap            Compress word gaps longer than N ms (default: 300, 0=off)
--cut-range          Manual cut range as start-end seconds (repeatable)
--margin             Safety margin around cuts in ms (default: 120)
--enhance            Enable Resemble Enhance neural denoising (off by default)
--cpu                Force CPU for Resemble Enhance (avoids MPS memory crashes)
--model              Whisper model (default: mlx-community/whisper-large-v3-turbo)
--min-confidence     Whisper confidence threshold (default: 0.5)
--output, -o         Output path (default: input_clean.mp4)
--verbose, -v        Verbose ffmpeg output
--json-output        Machine-readable JSON report
--keep-transcript    Save transcript JSON alongside output
```

## How it works

1. **Transcribe** — Whisper large-v3-turbo on Metal GPU via MLX. Word-level timestamps.
2. **Detect** — Pattern matching against filler word list with context-sensitive rules (e.g. "like" kept when semantic, "so" kept mid-sentence).
3. **Compress gaps** — Gaps exceeding threshold trimmed from the midpoint, preserving natural breath timing.
4. **Retake detection** — Sentence-level similarity comparison within a sliding window. Earlier takes get cut, latest kept.
5. **Cluster merge** — Adjacent cuts with < 200ms speech between them collapse into a single cut (prevents machine-gun jump cuts).
6. **Extract** — Frame-accurate segment extraction via ffmpeg filtergraph with 15ms audio fades.
7. **EQ** — Apply preset chain (highpass, bass, presence, compression, loudnorm).

Transcripts are cached by file hash. Re-runs skip transcription.

## License

MIT
