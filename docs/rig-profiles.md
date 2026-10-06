# Rig profiles

Use a rig profile to save equipment answers, sample measurements, and processing defaults.
Your profile lives at `<config dir>/rigs/<name>.json`.
On macOS and Linux, the config directory defaults to `~/.config/declip`.
Set `DECLIP_CONFIG_DIR` to use another directory.

## Create and select a profile

```bash
declip setup --name desk --sample sample.mp4 --preset voice --make-default
declip plan clip.mp4 --rig desk
declip rig list
declip rig show desk
```

Setup measures the middle sixty seconds, or the whole clip if shorter.
Your sample must last at least ten seconds.
Without a sample, measurements remain null and cannot qualify you for a measured tier.
Use `--mic`, `--room`, `--loudness`, `--destination`, `--nle-format`, and `--log-profile` to provide answers.
Use `--yes` to accept defaults for unanswered questions.
You get `usb`, `normal`, `-16`, `publish`, `edl`, and no log profile by default.
Setup does not infer a log profile from color tags alone.

An existing profile requires confirmation, or `--overwrite` without a terminal.
Use `declip rig rm desk` to remove a profile.
Without a terminal, removal requires `--force`.

## Understand your tier

You get the highest tier whose conditions match your answers and measurements.
Unknown measured values cannot satisfy a condition.

| Tier | Conditions |
|---|---|
| Pro | At least 10-bit video, ProRes video, a confirmed log profile, or at least 24-bit audio with an XLR interface |
| Creator | No pro condition; USB, lavalier, shotgun, or XLR microphone; measured noise below -55 dBFS |
| Basic | All other combinations |

| Setting | Basic | Creator | Pro |
|---|---|---|---|
| Publish destination | Render | Render | Render with match-source video policy |
| NLE destination | EDL or chosen FCPXML plus processed WAV | Same | EDL or chosen FCPXML; source media; no processing or WAV |
| No destination answer | Render | Render | EDL |
| Render video | H.264 8-bit, subject to the 10-bit protection gate | Match source | Match source; preserve 10-bit sources |
| Enhancer | `auto` | `none`; `auto` for echo/outdoor or noise above -50 dBFS | `none` |
| EQ | `voice` preset | Microphone high-pass | Empty |
| Loudness | Two-pass to your target | Same | None for NLE; two-pass for publish |
| Render AAC bitrate | 192 kbit/s | 256 kbit/s | 320 kbit/s |
| NLE WAV | PCM 24-bit at source rate | Same | None |
| Review | Required for every run | Required for every run | Required for every run |

EDL is the pro-tier default for NLE output.
FCPXML editor import remains unverified.
Normal CLI setup always supplies a destination, with `publish` as its default.
The unanswered pro EDL rule applies to profiles computed without a destination answer.

Creator high-pass values are 100 Hz for built-in microphones, 80 Hz for USB, and 70 Hz for lavalier.
Shotgun uses 90 Hz, and XLR uses 60 Hz.
Gap detection uses the noise floor plus six dB, clamped from -70 to -30 dBFS.
Without a measurement, it uses -55 dBFS.
Setup warns when it finds clipping.
Lower your input gain when the sample clips.

A new creator profile has noise below -55 dBFS.
Its condition for noise above -50 dBFS cannot arise from the same measurement.
The echo and outdoor conditions can still select `auto`.

## Read the schema

Rig schema version is 1.
Use `declip rig show desk` to inspect a complete generated profile before you edit it.

| Field | Contents |
|---|---|
| `schema_version` | `1` |
| `name`, `created` | Profile name and UTC creation time |
| `answers` | `mic`, `room`, `loudness`, `destination`, `nle_format`, `log_profile` |
| `measured` | Null or sample filename, window, audio/video facts, noise, loudness, peaks, and clipping |
| `hardware` | OS, architecture, chip, RAM, CPU cores, Apple-silicon flag, and GPU records |
| `tier`, `tier_reasons` | `basic`, `creator`, or `pro`, with reasons |
| `transcribe` | `backend`, `model`, `language`, `compute_type`, `device` |
| `audio` | `enhancer`, `enhancer_strength`, `eq_chain`, `loudness` |
| `video` | `encoder`, `codec`, `quality` |
| `output` | `mode`, `nle_format`, `sidecar_audio` |
| `detect` | `margin_ms`, `min_confidence`, `gap_noise_db`, `max_gap_ms`, `min_silence_ms`, `retakes` |
| `review` | `{"required": true}` |

A loudness object uses `i`, `tp`, and `lra`, for example `{"i": -16, "tp": -1.5, "lra": 11}`.
Use null to disable loudness.
Keep JSON booleans as `true` or `false`, and unknown values as `null`.

## Edit defaults

Back up the profile, then edit plain values inside `transcribe`, `audio`, `video`, `output`, or `detect`.
Keep its name, schema, and measured facts intact.
Reload it with `declip rig show desk` to check parsing and warnings.
A stored `review.required: false` becomes true with a warning.
It cannot bypass review.
Filter chains must pass the [security denylist](../SECURITY.md#validate-filter-chains).
A trailing `loudnorm` moves into the loudness target.
A second or nonterminal `loudnorm` fails validation.

You get values in this order: explicit command-line flags, rig profile, saved config defaults, then built-in defaults.
`--preset` replaces EQ and the preset loudness target.
An explicit `--loudness` takes precedence over a preset target.
A pro NLE setup profile keeps processing disabled even if you supply `setup --preset`.
Explicit run flags can override profile defaults.

The hardware record describes the machine that ran setup.
Runtime detection checks the current machine again.
A profile does not certify its new machine or force an unavailable backend to work.
Your edit list stores resolved profile and processing values.
Later profile edits do not change an existing edit list's audio chain.
