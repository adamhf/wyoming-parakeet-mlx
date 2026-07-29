# wyoming-parakeet

A [Wyoming protocol](https://github.com/rhasspy/wyoming) speech-to-text server
for Home Assistant, backed by NVIDIA's
[Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) running on
Apple Silicon via [parakeet-mlx](https://github.com/senstella/parakeet-mlx).

The model is loaded **in-process** — there is no HTTP hop between the Wyoming
bridge and inference.

## Why

On an M4 Mac mini, Parakeet transcribes a typical Home Assistant command in
**~100 ms** — roughly 5× faster than the best whisper.cpp configuration and
9× faster than whisper `large-v3`, while matching them on accuracy. See
[Benchmarks](#benchmarks) for the full comparison against eleven other
backends.

Two things matter beyond raw speed:

- **Silence returns an empty string.** Every whisper variant tested
  hallucinates on digital silence (`"Thank you."`, or `"you"` for
  faster-whisper), which reaches your conversation agent as a real utterance.
  Parakeet and Moonshine return nothing.
- **No cross-request contamination.** whisper.cpp's server carries decoder
  context between requests unless you pass `-nc`, and will return the
  *previous* utterance — in testing, roughly one time in five.

## Benchmarks

All figures measured on one machine: **M4 Mac mini (10-core, 32 GB), macOS 26.5**.
48 clips — 16 Home Assistant commands rendered through three macOS TTS voices
(Daniel, Samantha, Karen). Every backend loads once, transcribes all 48 clips
to warm up, then runs a timed pass. Latency is the **median** of that pass;
means are skewed by first-request kernel compilation.

Reproduce with `./bench/make_clips.sh && ./bench/benchmark.py --backend ...`.

| Backend | Runtime | Median | p90 | Exact | Digits | Silence |
|---|---|--:|--:|--:|--:|---|
| moonshine tiny | ONNX CPU | 28 ms | 44 ms | 31/48 | 15/21 | `""` |
| moonshine base | ONNX CPU | 53 ms | 70 ms | 35/48 | 20/21 | `""` |
| **parakeet-tdt-0.6b-v2** | **MLX** | **102 ms** | **112 ms** | **46/48** | **21/21** | `""` |
| parakeet-tdt-0.6b-v3 | MLX | 128 ms | 149 ms | 36/48 | 10/21 | `""` |
| faster-whisper tiny.en | CPU int8 | 182 ms | 204 ms | 44/48 | 21/21 | `"you"` |
| faster-whisper base.en | CPU int8 | 326 ms | 347 ms | 44/48 | 19/21 | `"you"` |
| whisper.cpp large-v3-turbo | Metal + CoreML | 518 ms | 537 ms | 47/48 | 21/21 | `"thank you"` |
| mlx-whisper large-v3-turbo | MLX | 828 ms | 848 ms | 47/48 | 21/21 | `"thank you"` |
| whisper.cpp large-v3 | Metal + CoreML | 941 ms | 1042 ms | 47/48 | 21/21 | `"thank you"` |
| faster-whisper small.en | CPU int8 | 974 ms | 1032 ms | 47/48 | 21/21 | `"you"` |
| mlx-whisper large-v3 | MLX | 1170 ms | 1251 ms | 48/48 | 21/21 | `"thank you"` |
| faster-whisper distil-large-v3 | CPU int8 | 4269 ms | 4304 ms | 46/48 | 21/21 | `"thank you"` |

**Exact** is a strict string match after normalising case, punctuation and
whitespace. **Digits** counts how many of the 21 number-bearing clips came
back with digits rather than spelled-out words — see
[Model choice](#model-choice-v2-not-v3) for why that matters more than it looks.
**Silence** is the output for three seconds of digital silence.

What the numbers say:

- **Parakeet v2 has the best latency/accuracy trade-off here.** It is 5×
  faster than the best whisper.cpp configuration and lands within one clip of
  it on accuracy.
- **`mlx-whisper large-v3` is the accuracy ceiling** — the only backend to
  score 48/48 — but costs 11× the latency to get there.
- **Moonshine is genuinely faster**, at 2× Parakeet's speed, and it also
  handles silence cleanly. It gives up real accuracy for it (35/48), so it is
  the right pick only if latency dominates everything else.
- **Parakeet v3's 10/21 on digits** is the ITN problem quantified. Its exact
  match (36/48) is dragged down almost entirely by that one behaviour.
- Both clips Parakeet v2 misses are the same word — "aircon", which the TTS
  voices render as "air con" / "aircan". Accuracy differences at the top of
  this table are concentrated in a couple of awkward tokens, not spread out.

### Caveats — read these before trusting the table

- **This is not a WER benchmark.** The audio is clean synthetic TTS from three
  similar English voices, with no noise, accents, crosstalk or far-field
  effects. It measures latency rigorously and accuracy only as a domain smoke
  test. For real word error rates see the
  [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).
- **faster-whisper is CPU-only on Apple Silicon.** CTranslate2 has no Metal
  backend, so those rows show CPU int8 performance. On an NVIDIA GPU they
  would look completely different — do not read this as a verdict on
  faster-whisper generally, only on what it does on this hardware.
- **Latency is raw inference**, excluding Wyoming protocol overhead. End to
  end through this server, expect roughly 15–35 ms on top.
- One machine, one run each. Treat differences of a few percent as noise.

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
onto your LLM fallback.

Measured on the benchmark corpus, v3 returned digits for only **10 of 21**
number-bearing commands, against **21/21** for v2. That single behaviour is
most of the gap between their exact-match scores. v3 remains the right choice
if you need languages other than English — just size the trade-off first.

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
