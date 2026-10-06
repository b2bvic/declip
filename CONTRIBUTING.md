# Contribute to declip

Use a branch for each change.
Read the current code and frozen contracts before editing.
Keep changes within the assigned file boundaries.
Add a regression test when behavior changes.
Describe test evidence and known limits in your pull request.

## Prepare your environment

Install uv, Python 3.11 or later, ffmpeg, and ffprobe.
CI uses Python 3.11 on `ubuntu-latest` and `macos-latest` only.
Windows remains planned and untested.

```bash
uv sync --locked --extra dev
export DECLIP_CONFIG_DIR="$(mktemp -d)"
export DECLIP_CACHE_DIR="$(mktemp -d)"
uv run --locked --extra dev pytest -q --require ffmpeg --require libx265
uv run --locked --extra dev pytest -q tests/contracts
uv run --locked --extra dev ruff check src tests scripts
```

Remove your temporary state directories after testing.
Tests also isolate config and cache for each case.
Do not read private settings, media, browser sessions, or live write endpoints in tests.

## Use fixtures and required capabilities

Use synthetic transcript text only.
Use the fake transcriber for CLI tests with `DECLIP_TRANSCRIBER=fake`.
Set `DECLIP_FAKE_TRANSCRIPT_DIR` to your fixture directory.
The fixture filename must match the source stem.
The automatic backend selector never chooses fake.
Do not download model weights in tests.

Generate small media with ffmpeg lavfi sources in `tmp_path`.
Mark media tests `ffmpeg`.
Use software codecs such as `libx264`, `libx265`, AAC, or PCM.
Delete generated media when the test or temporary context finishes.
Do not commit media files.

Repeat `--require NAME` to turn a required capability's skip into a failure.
Supported names include `ffmpeg`, `libx265`, `cpu`, and `videotoolbox`.
The standard dev environment can skip the optional CPU-extra availability test.
Install the CPU extra to require that check without fetching weights:

```bash
uv run --locked --extra dev --extra cpu pytest -q tests/test_transcribe.py --require cpu
```

Hosted CI excludes `hwenc` tests and does not prove NVIDIA or VideoToolbox hardware.
Keep hardware receipts separate from ordinary unit-test claims.

## Review golden changes

Author expected golden files from fixture text and documented timing rules.
Do not generate expectations by calling the implementation under test.
`pytest --update-golden` writes failing output to `<golden>.actual` only.
It never replaces the golden file.
Inspect the diff and edit the expected file by hand when the change is correct.
Require review of that change before promotion.

## Audit runtime dependencies

```bash
uv run --locked --extra dev python scripts/audit.py --extra cpu --platform macos
uv run --locked --extra dev python scripts/audit.py --extra cpu --platform linux
uv run --locked --extra dev python scripts/audit.py --extra cuda --platform linux
uv run --locked --extra dev python scripts/audit.py --extra mac --platform macos
```

Each command resolves Python 3.11 runtime requirements and scans them without installing the extra.
The script writes receipts in `security/audit/` and updates `security/dependency-audit.json`.
A finding, empty resolution, skipped package, or scan error fails the command.
The index stays held until every supported combination has a current clean receipt.
Compilation resolves `pyproject.toml`; it does not export locked versions from `uv.lock`.
The Mac extra targets macOS 14, matching the locked MLX and Torch wheel minimum.
The CPU extra targets macOS 13.
The receipt records the deployment target.
Do not describe these scans as audits of every locked dependency.
CI repeats each audit on its native runner and saves its receipt as an artifact.

Dependabot tracks GitHub Actions and the `uv` ecosystem.
GitHub lists `uv` in its [supported ecosystem table](https://docs.github.com/en/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories).
Keep every workflow action pinned to a forty-character commit SHA.
Keep top-level workflow permissions at `contents: read`.
Run `tests/test_workflows.py` before changing CI.
Version 0.5.1 has no package publication workflow.

## Document evidence

Label Linux with NVIDIA as assumed until a real hardware receipt passes.
Keep FCPXML editor import unverified until you have an editor-import receipt.
Keep the synthetic prompt result inconclusive until real-speech evidence resolves it.
Require review for every run and every tier.
Use second person, active voice, short sentences, and sentence-case headings.
