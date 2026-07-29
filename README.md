# wyoming-parakeet

A [Wyoming protocol](https://github.com/rhasspy/wyoming) speech-to-text server
for Home Assistant, backed by NVIDIA's
[Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) running on
Apple Silicon via [parakeet-mlx](https://github.com/senstella/parakeet-mlx).

The model is loaded **in-process** — there is no HTTP hop between the Wyoming
bridge and inference.

## Why

Measured on an M4 Mac mini against ten typical Home Assistant voice commands,
replacing a whisper.cpp setup:

| Backend | Mean latency | Correct | Silent input |
|---|---|---|---|
| whisper.cpp `large-v3` | ~1150 ms | 10/10 | `"Thank you."` |
| whisper.cpp `large-v3-turbo` | ~570 ms | 10/10 | `"Thank you."` |
| **parakeet-tdt-0.6b-v2** | **~110 ms** | **10/10** | `""` |

Two things matter beyond raw speed:

- **Silence returns an empty string.** Whisper hallucinates `"Thank you."` on
  digital silence, which reaches your conversation agent as a real utterance.
- **No cross-request contamination.** whisper.cpp's server carries decoder
  context between requests unless you pass `-nc`, and will return the
  *previous* utterance — in testing, roughly one time in five.

## Requirements

- Apple Silicon Mac (MLX is Metal/ANE-backed)
- Python 3.10+ **with the `lzma` module** — `librosa` pulls in `pooch`, which
  imports it. Pythons built without `xz` (a common pyenv default) pass every
  version check and then fail at import time with
  `ModuleNotFoundError: _lzma`. Homebrew's Python is fine.
- ~2.3 GB disk for the model, ~600 MB for MLX wheels

`ffmpeg` is **not** required — Wyoming already delivers 16 kHz mono PCM, so
the mel spectrogram is built directly.

## Install

```bash
git clone https://github.com/adamhf/wyoming-parakeet-mlx
cd wyoming-parakeet-mlx
./install.sh
```

This creates a virtualenv, runs the unit tests, pre-downloads the model, and
registers a LaunchDaemon on port 7892 that starts at boot without needing a
GUI login. It installs *in place*, so keep the checkout somewhere permanent.

Options: `--port`, `--model`, `--user`, `--python`, `--no-daemon`,
`--no-download`.

Then in Home Assistant: **Settings → Devices & Services → Add Integration →
Wyoming Protocol**, enter the host and port, and select the new engine as the
speech-to-text step of your Assist pipeline.

Remove it with `./uninstall.sh`.

## Model choice: v2, not v3

The default is `parakeet-tdt-0.6b-v2` even though v3 is newer and
multilingual, because of inverse text normalisation:

| Spoken | v2 | v3 |
|---|---|---|
| "twenty one degrees" | `21 degrees` | `twenty-one degrees` |
| "thirty percent" | `30%` | `thirty percent` |

Home Assistant's local intent matching (hassil) expects digits. If your
pipeline has `prefer_local_intents` enabled, a model that spells numbers out
still *looks* accurate while quietly pushing commands off the fast local path
onto your LLM fallback. v3 is the better choice if you need languages other
than English — just be aware of the trade.

## Updating the model

`HF_HUB_OFFLINE=1` is set in the daemon, so it never silently re-downloads or
changes model at boot, and starts fine without a network. Updating is
therefore deliberate:

```bash
HF_HUB_OFFLINE= .venv/bin/python -c \
  "from parakeet_mlx import from_pretrained; from_pretrained('mlx-community/parakeet-tdt-0.6b-v3')"
```

Then re-run `./install.sh --model mlx-community/parakeet-tdt-0.6b-v3`. The old
model stays cached, so reverting is just another `./install.sh`.

## Tests

```bash
.venv/bin/python -m pytest
```

37 unit tests, well under a second. The model is mocked throughout, so they
need no GPU, no network and no 2.3 GB download — they cover the audio
marshalling and threading around it, which is where the real bugs were. Every
regression test was mutation-checked: the fix reverted, the test confirmed to
fail.

Worth knowing about a few:

- `test_load_and_inference_share_one_thread` — MLX streams are thread-local,
  so the model must be loaded *and* evaluated on the same thread or `mx.eval()`
  raises `There is no Stream(cpu, 1) in current thread`. The test deliberately
  *overlaps* its calls: sequential calls can coincidentally reuse one thread
  out of a multi-worker pool and pass a broken implementation.
- `test_audio_is_float32_not_bfloat16` — `get_logmel` views the complex STFT
  output as the input dtype, so anything narrower silently doubles the mel bin
  count and the matmul fails.
- `test_model_failure_still_sends_a_transcript` — Wyoming's run loop is
  `try/finally` with no `except`, so an exception escaping `handle_event`
  closes the connection having sent nothing, and Home Assistant waits for a
  response that never arrives. The handler catches and returns an empty
  transcript so it fails fast instead.
- `test_concurrent_handlers_do_not_share_audio` — guards against
  reintroducing the cross-request contamination described above.

### End to end

Unit tests never touch the real model, so after any model or library change:

```bash
./test/make-clips.sh                                    # generates via macOS TTS
.venv/bin/python test/wy-test.py test/clips/*.wav
```

Expect all ten commands correct, `silence.wav` empty, ~100–150 ms each once
warm. The first request or two after a restart run slower (~200–350 ms) while
Metal compiles its kernels. **Check number formatting, not just the words** —
`cmd2`, `cmd4` and `cmd7` are the ones that catch a model with weak ITN.

## Updating the library

```bash
.venv/bin/pip install -U parakeet-mlx mlx mlx-metal wyoming
```

Re-run both test suites afterwards. This project calls `get_logmel()` directly
rather than `load_audio()` (which shells out to ffmpeg), so it depends on two
parakeet-mlx internals rather than public API — the tests above are what tell
you if either moved.

## Logs

```bash
tail -f /tmp/local.wyoming-parakeet.stderr
```

Each request logs audio duration, inference time and the transcript.

## License

MIT
