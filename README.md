# Local video filler word removal: declip

Declip plans local video cuts for creators and editing teams. Use its filler and gap heuristics to review talking-head edits before processing media.

[Project page](https://scalewithsearch.com/code/declip)

## Install

Requirements: Python 3.11 or later.

```bash
gh repo clone b2bvic/declip
cd declip
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
```

This installs the portable test environment. Full processing requires Apple Silicon, uv, ffmpeg, and the declared runtime dependencies.

## Quick start

```bash
.venv/bin/python declip --help
.venv/bin/python -m pytest -q
```

## How it works

- Transcribe media with MLX Whisper when the full runtime is installed.
- Build cut regions from filler, gap, retake, and manual-range rules.
- Use ffmpeg for requested processing.

## Limits

- The full runtime remains held for dependency review; see SECURITY.md and security/dependency-audit.json.
- Portable tests cover cut calculations without downloading transcription weights.
- Filler and retake detection are heuristic.
- The current filler detector does not enforce its minimum-confidence argument.

## Related repositories

- [web2md](https://github.com/b2bvic/web2md)
- [twitter-bookmarks](https://github.com/b2bvic/twitter-bookmarks)
- [sws-skills](https://github.com/b2bvic/sws-skills)

## Development

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check --select E9,F63,F7,F82 declip tests
```

CI runs the portable tests and checks syntax-related Python lint rules.

## License

MIT. See [LICENSE](LICENSE).
