# Models and prompt behavior

Choose `mlx` for Apple silicon on macOS 14 or later or `faster` for CPU and Linux CUDA transcription.
An explicit unavailable backend or device fails instead of silently switching.
The core install does not include a model library.
Use the `mac`, `cpu`, or `cuda` install extra for transcription.

## Choose a default model

Setup detects hardware and records a default model:

| Hardware | Backend | Model | Compute type |
|---|---|---|---|
| Apple silicon with at least 16 GB RAM | MLX | `large-v3-turbo` | `auto` |
| Apple silicon with less RAM or unknown RAM | MLX | `small` | `auto` |
| NVIDIA with at least 6 GB VRAM | faster | `large-v3-turbo` | `float16` |
| NVIDIA with less or unknown VRAM | faster | `large-v3-turbo` | `int8_float16` |
| CPU with at least eight cores | faster | `large-v3-turbo` | `int8` |
| CPU with fewer cores | faster | `small` | `int8` |

Treat these thresholds as defaults, not measured speed guarantees.
Linux with NVIDIA remains assumed, not verified.
Windows is planned for a later release and is untested.
Pass `--model small` to choose another supported model.

## Map model names

P3 verified the following repository names and total sizes through hub metadata on 2026.10.06.
Sizes include all repository files.
The downloader requests backend load files only, so your download can be smaller.

| Short name | MLX repository | Bytes | faster repository | Bytes |
|---|---|---:|---|---:|
| `small` | [mlx-community/whisper-small-mlx](https://huggingface.co/mlx-community/whisper-small-mlx) | 481,309,720 | [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) | 486,215,847 |
| `large-v3` | [mlx-community/whisper-large-v3-mlx](https://huggingface.co/mlx-community/whisper-large-v3-mlx) | 3,083,522,487 | [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | 3,090,839,273 |
| `large-v3-turbo` | [mlx-community/whisper-large-v3-turbo](https://huggingface.co/mlx-community/whisper-large-v3-turbo) | 1,613,979,758 | [mobiuslabsgmbh/faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) | 1,621,668,947 |

The assumed names `mlx-community/whisper-small` and `mlx-community/whisper-large-v3` act as compatibility aliases.
The published names include the `-mlx` suffix.
An MLX name maps to its matching short name on the faster backend.
You can also pass a local CTranslate2 model directory with `--backend faster --model /path/to/model`.

## Understand the filler prompt

The English filler file starts with this prompt:

```text
# prompt: Um, uh, hmm, like, you know, I mean. So, uh, yeah.
```

With automatic language detection, transcription first checks thirty seconds without a prompt.
It then uses the detected language and that language's filler prompt for the full pass.
Only English filler rules ship.
For another language, supply `<config dir>/fillers/<language>.txt` with your own prompt and filler entries.
Without a matching filler file, you get gap and retake proposals only.

The synthetic filler-prompt probe was inconclusive for both backends.
No real-speech receipt exists because the intended clip's local bytes were unavailable.
Prompt persistence after thirty seconds remains unproven.
Isolated synthetic fillers do not establish recognition of short hesitations joined to speech.

| Backend | Initial prompt, first thirty seconds | Initial prompt, later | No prompt, first/later | Finding |
|---|---|---|---|---|
| MLX on Metal | 6/6 | 12/12 | 6/6 and 12/12 | Inconclusive |
| faster on CPU | 4/6 | 12/12 | 6/6 and 12/12 | Inconclusive; one false hit with prompt |

`--prompt-mode auto` therefore uses `initial` on both backends.
Choose `chunked` to repeat the prompt in each thirty-second chunk.
On faster, you can also choose `hotwords`.
Mocked tests cover those argument paths; they do not prove recognition quality.
Review is required for every run and every tier.
A prompt can also create a filler that is absent from your audio.
Check the audio before accepting a proposal.

## Control downloads

Before the first download, you see the repository ID, source host, cache directory, and reported size.
If metadata fails, you see `size unknown`.
Model weights use the Hugging Face cache, separate from the declip transcript cache.
Use `HF_HOME` or `HF_HUB_CACHE` to set the hub location.
Use `DECLIP_CACHE_DIR` to set the declip cache location.

For an offline run:

```bash
HF_HUB_OFFLINE=1 declip plan clip.mp4 --model small
```

Cache the selected model first.
If the weights are missing, the command fails and names the model and `HF_HUB_OFFLINE`.
Offline mode makes no hub network request.
Local model directories bypass the hub.

## Consider other models

[CrisperWhisper](https://huggingface.co/nyrahealth/faster_CrisperWhisper) supports English and German under CC-BY-NC-4.0.
Check that noncommercial license against your intended use.
The app does not select or download it.
You can supply your own compatible local CTranslate2 directory.
The app does not include a whisper.cpp backend.
