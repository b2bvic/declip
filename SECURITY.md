# Security policy

Report vulnerabilities through the repository's private GitHub security reporting channel when available.
Otherwise, contact the maintainer through the GitHub profile.
Keep credentials, personal records, and exploit details out of public issues.

## Read runtime audit results

The 2026.10.06 scans used pip-audit 2.10.1 with Python 3.11 resolutions.
The overall release verdict is **clean** for the four supported audit targets.
The Mac extra targets macOS 14, matching the locked MLX and Torch wheel minimum.
The CPU extra targets macOS 13.
Older macOS versions are outside the Mac-extra audit claim.

| Extra | Target | Result | Advisory records | Receipt |
|---|---|---|---:|---|
| `cpu` | macOS 13 arm64 | Clean | 0 | [cpu-macos.json](security/audit/cpu-macos.json) |
| `cpu` | Linux x86_64 | Clean | 0 | [cpu-linux.json](security/audit/cpu-linux.json) |
| `cuda` | Linux x86_64 | Clean | 0 | [cuda-linux.json](security/audit/cuda-linux.json) |
| `mac` | macOS 14 arm64 | Clean | 0 | [mac-macos.json](security/audit/mac-macos.json) |

Read the [audit index](security/dependency-audit.json) for scope and timestamps.
These scans resolve runtime requirements from `pyproject.toml`, not from a lockfile export.
They do not audit the development environment or the standalone enhancer binary.
A clean result means no known advisory in that resolved graph at scan time.
It does not prove absence of vulnerabilities or native hardware support.
Review dependency fixes and rerun all supported audits before marking the index clean.
CI fails when an audit finds an advisory or cannot scan every resolved package.

Windows has no audit in 0.5.0 because support is planned for a later release and is untested.
Linux with NVIDIA remains assumed, not verified, even though its dependency scan is clean.

## Understand the review server

Review is required for every run and every tier.
Render and export refuse pending review or changed source bytes.
The server binds only to `127.0.0.1` and serves one edit list and one preview file.
Every route needs the session token, including static assets and media.
Keep the printed token URL private.

The server checks Host to limit DNS rebinding.
It checks a supplied Origin on write requests and requires JSON content.
It sends no CORS headers.
Static assets come from a fixed filename map.
The browser cannot request an arbitrary local path.
Transcript text becomes text nodes instead of HTML.
Stale revisions fail instead of silently overwriting another writer's changes.

These checks protect the local browser workflow.
They do not isolate you from a hostile local process that can read your files or tokens.
Use SSH forwarding with matching ports for remote review.
Do not expose the loopback service through a public proxy.

## Validate filter chains

Shared edit lists and presets can contain untrusted ffmpeg filter text.
The app rejects `movie`, `amovie`, `sendcmd`, `asendcmd`, `ladspa`, and `lv2`.
It also rejects `metadata` or `ametadata` when they have a `file=` option.
Every chain from presets, CLI flags, profiles, and edit lists passes this check before use.
You cannot put multiple `loudnorm` filters in a chain or put one before the last filter.

The denylist limits known file and command surfaces.
It does not sandbox ffmpeg or prove that every permitted filter is safe.
Inspect shared processing settings before using them.

## Understand the review hash

The source uses full-file SHA-256 binding.
The review hash covers the source hash and cut decisions.
This hash is a workflow guard against stale review, not a security control.
Anyone who can edit the JSON can recompute it.
It is not a signature or proof of the reviewer's identity.
Keep your source, edit lists, model files, and config under your own access controls.
