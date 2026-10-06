# Platform setup

Use macOS or Linux for version 0.5.0.
Windows is planned for a later release and is untested.
Linux with NVIDIA is assumed, not verified.
No Linux hardware receipt exists.
Local macOS tests cannot prove Linux CUDA loading or NVENC output.

## Install media tools

On macOS, install `ffmpeg` through Homebrew:

```bash
brew install ffmpeg
```

On Ubuntu or Debian:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

For other Linux distributions, use your package manager.
Confirm both tools:

```bash
ffmpeg -version
ffprobe -version
declip doctor --json
```

Your build must provide `libx264`, `libx265`, and an AAC encoder for software video renders.
Doctor exits 1 when a tool or transcription backend is unavailable.
A core-only editing install can still review, render, and export existing plans.

## Use Apple silicon

Install the `mac` extra for MLX transcription on macOS 14 or later.
The locked MLX and Torch wheels require that minimum version.
The Mac-extra audit uses the same macOS 14 deployment target.
You get software decode unless a source-specific hardware decode probe passes.
A listed hardware encoder must also pass an encode probe before selection.
The local render test verified HEVC Main 10 through VideoToolbox.
Your output also passes a codec and bit-depth check after rendering.

## Use Linux with NVIDIA

Install the `cuda` extra.
You need a compatible NVIDIA driver, CUDA 12 libraries, and cuDNN 9 libraries.
The extra supplies cuBLAS and cuDNN wheels.
The loader opens `libcublas.so.12` and `libcudnn*.so.9` from their installed package directories before importing CTranslate2.
This preload path remains unverified on Linux hardware.
Check `nvidia-smi` and `declip doctor --json` before your first run.
An explicit `--device cuda` fails if CUDA is unavailable.
Use `--device cpu` with a CPU backend when you choose to run on CPU.

Windows CUDA DLL loading is outside version 0.5.0.
Do not interpret wheel availability as Windows support.

## Review a remote run

On the remote machine:

```bash
declip review clip.mp4 --no-browser --port 8765
```

On your local machine:

```bash
ssh -L 8765:127.0.0.1:8765 your-host
```

Open the printed `http://127.0.0.1:8765/?token=...` URL locally.
Keep the same port at both ends so the Host check passes.
Keep the token private and leave SSH connected until you finish review.
The server binds only to loopback.
You cannot select a public bind address.

## Choose an enhancer

Use `none` to preserve PCM without denoising.
Use `afftdn` for the built-in ffmpeg denoiser.
With `auto`, you get `deepfilter` if available, otherwise `afftdn`.
An explicit unavailable `deepfilter` choice fails.

You must install the standalone `deep-filter` binary yourself.
The app does not download it.
Put it on PATH as `deep-filter`, or set `deepfilter_path` in your `config.json`.
P6 inspected the [DeepFilterNet v0.5.6 assets](https://github.com/Rikorose/DeepFilterNet/releases/tag/v0.5.6) on 2026.10.06.

| Platform | Standalone release asset | Evidence |
|---|---|---|
| macOS arm64 | `deep-filter-0.5.6-aarch64-apple-darwin` | Asset and local stereo probe verified |
| macOS x86_64 | `deep-filter-0.5.6-x86_64-apple-darwin` | Asset verified; execution unverified |
| Linux x86_64 | `deep-filter-0.5.6-x86_64-unknown-linux-musl` | Asset verified; execution assumed |
| Linux aarch64 | `deep-filter-0.5.6-aarch64-unknown-linux-gnu` | Asset verified; execution assumed |
| Linux armv7 | `deep-filter-0.5.6-armv7-unknown-linux-gnueabihf` | Asset verified; execution assumed |
| Windows | Release asset exists | Unavailable in 0.5.0; planned and untested |
| Other combinations | No selected asset | Unavailable; use `afftdn` |

Strength maps from zero to one onto `--atten-lim-db` from zero to one hundred dB.
The enhancer restores your source sample rate, channels, and sample count after processing.
The upstream neural binary uses a 16-bit intermediate.
A final 24-bit WAV cannot recover precision lost inside that binary.

## Check verification evidence

Hardware certification requires separate platform receipts.
Use the [README evidence matrix](../README.md#check-platform-support) for the current stated limits.
GitHub Actions runs the test suite on `ubuntu-latest` and `macos-latest` for each push. These runners have no GPU.
