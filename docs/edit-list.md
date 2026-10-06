# Edit-list reference

Planning writes `<source stem>.declip.json` next to your source.
Use `--edit-list /path/to/list.json` to choose another location.
The JSON describes cuts and processing without rewriting your media.
Render and export consume the same file after review.

## Read schema 3

| Field | Contents |
|---|---|
| `schema_version` | `3` |
| `declip_version` | `0.5.0` for newly planned lists |
| `source` | Resolved `path`, filename `name`, byte `size`, full-file `sha256` |
| `media` | Duration, start time, container, selected stream indices, frame rate, VFR, dimensions, rotation, codec, bit depth, color, audio streams, dropped streams |
| `rig` | Name, tier, and resolved profile snapshot; null name/tier if absent |
| `transcript` | Null or backend, model, language, prompt, words, and segments |
| `cuts` | Proposed, accepted, rejected, and manual cut rows |
| `processing` | Enhancer, strength, EQ chain, loudness target, crossfade milliseconds |
| `output` | `mode`, `nle_format`, `sidecar_audio` |
| `review` | State, mode, pass time, and decision hash |
| `history` | Stage records with timestamps and migration metadata |

All times use seconds from the first decoded source sample.
Frame rates use rational strings such as `30000/1001`.
Color tags use `primaries`, `trc`, `matrix`, and `range`.
Audio stream records use index, codec, rate, channels, layout, bit depth, and language.

A transcript word has `i`, `start`, `end`, `text`, `p`, and `segment`.
Word indices start at zero and remain contiguous.
A segment has `i`, `start`, `end`, `text`, and `word_indices`.

A cut row contains:

```json
{
  "id": "9f2c0a1b2c3d4e5f",
  "kind": "filler",
  "start": 12.31,
  "end": 12.74,
  "label": "um",
  "confidence": 0.88,
  "low_confidence": false,
  "words": [41, 41],
  "origin": "auto",
  "status": "proposed"
}
```

You can use kinds `filler`, `gap`, `retake`, and `manual`.
Statuses are `proposed`, `accepted`, and `rejected`.
`words` holds inclusive word indices, or null for gaps and manual cuts.
Manual cuts have origin `manual` and status `accepted`.
They still require you to finish review.
A low-confidence badge does not decide the cut.
IDs derive from kind, six-decimal boundaries, and label.
Keep IDs unique within the file.

## Finish review

Review is required for every run and every tier.
Pending review has `state: "pending"` and null mode, pass time, and hash.
A completed review has `state: "passed"` and `mode: "interactive"`.
You must decide every proposed cut before finishing, including low-confidence proposals.
For a zero-cut plan, select **Finish review** explicitly.

The decision hash covers the source SHA-256 and each cut's ID, kind, boundaries, and status in file order.
Serialization sorts object keys and uses compact JSON separators.
Any decision change makes review pending again.
Render and export require the stored hash to match current decisions.
They also compute a fresh full-file source hash.
A changed source fails even if its size and cached metadata match.

The review hash is a workflow guard, not a security control.
Anyone who can edit the JSON can recompute it.
Use only edit lists whose source and processing you trust.

## Rerun planning

```bash
declip plan clip.mp4 --stages fillers,gaps,retakes
```

A rerun replaces automatic cuts for the selected stages only.
It preserves statuses by deterministic cut ID and keeps manual cuts.
Review stays passed only if the final source-bound decision hash remains unchanged.
Use `--reset` to discard automatic decisions and manual cuts and start a new review.
A changed source requires `--reset` for a new plan.

Render and exports use accepted cuts only.
They clamp and merge removed intervals, then snap boundaries to frames or audio samples.
They preserve every positive keep on that grid.
This shared timeline also controls captions, markers, duration reports, and browser preview.
If you remove all media, review can finish, but render and export refuse the empty result.

## Migrate schema 2

```bash
declip migrate clip.edit.json
```

Migration writes `clip.declip.json` and leaves the original bytes unchanged.
Other filenames gain `.schema3.json`.
Use `--out` to choose the destination or `--source` to relocate the source.
Migration refuses a missing or changed source and an existing destination.
It does not detect cuts or silently approve decisions.
You must review the migrated list.

Migration changes `vetoed` to `rejected` and flattens old transcript words.
It resolves the old preset into an EQ chain and loudness target.
Unknown presets warn and use the built-in raw loudness target.
Old LUT and caption settings survive as inert history only.
Review always resets to pending.
An old schema-2 file cannot pass directly to plan, render, or export.

## Keep exports separate

Export names use `<stem>.declip.edl`, `.fcpxml`, `.srt`, `.markers.json`, and `.wav` for sidecar audio.
No export writes the reserved `.declip.json` edit-list name.
EDL is the pro-tier default for NLE output.
FCPXML import into an editor remains unverified.
For variable-frame-rate video, EDL and FCPXML refuse and show a conform command.
Plan and review the conformed file again before exporting it.
