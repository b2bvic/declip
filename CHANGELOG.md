# Changelog

## 0.5.1

### Fix real-footage results

- Report real-speech prompt probes as passed, failed, or inconclusive in the receipt.
- A no-prompt baseline with zero fillers is inconclusive and leaves the synthetic prompt-mode decision unchanged.
- Keep transcription backend diagnostics off stdout in JSON modes, including language detection. Human-readable output stays unchanged.
- Update versioned Git install commands to `v0.5.1`.

## 0.5.0

### Change your install and workflow

- Install from the versioned Git URL with uv instead of running the repository script.
- Select the `mac`, `cpu`, or `cuda` extra for transcription. The core install supports editing without model libraries.
- Create named rig profiles with `declip setup`, then select them with `--rig`.
- Plan cuts, review them in your local browser, then render or export.
- Review is required for every run and every tier. Flags and profiles cannot bypass it.
- `process --execute` and `clean --execute` now write the edit list and stop for review.
- The older `--export` option also stops at the edit list. Run review and export separately.
- Place compatibility options before or after subcommands. Subcommand values take precedence.

### Update saved settings

- The old personal default preset is removed. Without a rig or preset choice, you get `raw` loudness normalization only.
- Choose `voice`, `natural`, `podcast`, `raw`, or `none`, or keep your own user preset.
- An unknown saved preset warns and falls back to `raw` without rewriting your config.
- User `presets.json` entries override shipped entries by name.
- The legacy `fillers.txt` still overrides the English filler set. Language-specific files can also include a prompt.
- A trailing preset `loudnorm` becomes the target for one two-pass normalization stage.
- `--enhance` selects standalone DeepFilterNet if available, otherwise ffmpeg denoising. The old Python enhancement graph is removed.
- `--cpu` remains accepted with no effect and a deprecation note. Use `--device cpu`.
- Compatible MLX model names map to faster backend model names. Published small and large-v3 MLX names include `-mlx`.

### Preserve media and decisions

- Render matches your source by default. Use `--codec h264` when you explicitly need H.264.
- A 10-bit source cannot become 8-bit without `--allow-8bit` and a compatible codec choice.
- Render batches video, uses one source-PCM audio chain, and snaps cuts to the shared frame or sample grid.
- Render keeps audio channels and sample rates, cuts other audio tracks, and preserves global metadata.
- Subtitle, data, attachment, and timecode streams are dropped.
- Output guards check codec, bit depth, and duration. Existing outputs require `--overwrite`; source aliases are refused.
- Schema-3 edit lists use full source hashes and preserve decisions by cut ID on reruns.
- Use `declip migrate` for schema-2 lists. Migration leaves the old file unchanged and resets review.
- Export names now use `<stem>.declip.edl`, `.fcpxml`, `.srt`, and `.markers.json`.
- SRT uses retained words on the output timeline. Exports never overwrite your edit list.
- EDL is the pro-tier default for NLE output. FCPXML remains an explicit choice.
- Low-confidence proposals carry a badge and still need your decision.
- JSON modes keep stdout limited to one JSON document. Diagnostics use stderr.

### Check evidence and limits

- Windows is planned for a later release and is untested.
- Linux with NVIDIA is assumed, not verified.
- The synthetic filler-prompt probe is inconclusive on both backends. No real-speech receipt exists.
- FCPXML passed structural checks, but import into an editor is unverified.
- CI runs on `ubuntu-latest` and `macos-latest` only.
- Runtime audits cover four supported extra/platform combinations. See [security results](SECURITY.md) for each scan and its deployment target.
- Caption burn-in, LUT processing, forced 4K export, Windows support, and PyPI publication remain outside this version.
