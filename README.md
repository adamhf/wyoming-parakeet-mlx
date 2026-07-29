# wyoming-parakeet-mlx

A [Wyoming protocol](https://github.com/rhasspy/wyoming) speech-to-text server
for Home Assistant, backed by NVIDIA's
[Parakeet TDT](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) running on
Apple Silicon via [parakeet-mlx](https://github.com/senstella/parakeet-mlx).

The model is loaded **in-process** — there is no HTTP hop between the Wyoming
bridge and inference.

## Why

Speed is the obvious reason — see [Performance](#performance) below — but two
behaviours matter just as much for a voice assistant:

- **Silence returns an empty string.** Every whisper variant tested
  hallucinates on digital silence (`"Thank you."`, or `"you"` for
  faster-whisper), which reaches your conversation agent as a real utterance.
  Parakeet and Moonshine return nothing.
- **No cross-request contamination.** whisper.cpp's server carries decoder
  context between requests unless you pass `-nc`, and will return the
  *previous* utterance — in testing, roughly one time in five.

## Performance

On an M4 Mac mini, against 48 clips of Home Assistant commands:

| Backend | Median | Exact | Silence |
|---|--:|--:|---|
| **parakeet-tdt-0.6b-v2** (this) | **102 ms** | **46/48** | `""` |
| whisper.cpp large-v3-turbo | 518 ms | 47/48 | `"thank you"` |
| whisper.cpp large-v3 | 941 ms | 47/48 | `"thank you"` |
| faster-whisper base.en | 326 ms | 44/48 | `"you"` |

Roughly 5× faster than the best whisper.cpp configuration, matching it within
one clip on accuracy. **[Full comparison against 11 backends, methodology and
caveats → BENCHMARKS.md](BENCHMARKS.md)**, including Moonshine (faster still,
meaningfully less accurate) and mlx-whisper (the accuracy ceiling, 11× the
latency).

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

### Server options

`install.sh` bakes these into the plist; run `script/run --help` to see them
all. The ones worth knowing:

| Flag | Default | |
|---|---|---|
| `--uri` | — | `tcp://0.0.0.0:7892`. Bind to one interface to limit exposure. |
| `--model` | `…parakeet-tdt-0.6b-v2` | See [Model choice](#model-choice-v2-not-v3). |
| `--max-audio-seconds` | `120` | Cap on buffered audio per utterance. |
| `--debug` | off | Verbose logging, **including transcript text**. |

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

44 unit tests, well under a second. The model is mocked throughout, so they
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
- `test_audio_buffer_is_capped` and `test_transcript_text_is_not_logged_at_info`
  — the two [security](#security) properties. Both are easy to regress with an
  innocent-looking refactor and invisible when they break.

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

Each request logs audio duration, inference time and the transcript's
character count — **not the transcript itself**. Pass `--debug` to include the
text, bearing in mind that the log then contains everything said to your voice
assistant, in a file other local accounts can read. See [Security](#security).

The daemon runs as its configured `--user`, and launchd opens these files as
that user; if you change `--user`, re-run `install.sh` rather than editing the
plist, so the existing log files are chowned across.

## Security

**The Wyoming protocol has no authentication or transport encryption.** Any
client that can reach the port can submit audio, and Home Assistant trusts
whatever this service returns — a transcript becomes an intent, and an intent
unlocks doors. Treat the port as a trust boundary and keep it on a network you
control.

Two things this server does about that:

- **Buffered audio is capped** (`--max-audio-seconds`, default 120). Audio
  accumulates until `AudioStop`, and a client that never sends one — malicious
  or just stuck — otherwise grows the buffer indefinitely. Measured at ~11 MB/s
  over loopback, enough to exhaust 32 GB in under an hour from one connection.
  Over the cap, further audio is dropped with a single warning and whatever was
  captured is still transcribed.
- **Transcripts are not logged at INFO.** The log records duration, latency and
  character count; the text itself is behind `--debug`. Log files are
  long-lived, and on macOS `/tmp` they are world-readable by default — meaning
  every voice command would otherwise sit in plaintext readable by any local
  account.

Worth doing yourself, depending on your threat model:

- **Bind to one interface.** The daemon listens on `0.0.0.0`, so it is exposed
  on every network the host is attached to — including VPN interfaces like
  Tailscale, which is easy to overlook. Set the URI to a specific address
  (`--uri tcp://192.168.1.10:7892`) or firewall the port to your Home
  Assistant host.
- **Do not run it as an admin account.** The installer defaults `--user` to
  whoever runs it. On a typical macOS setup that account is in `admin`, and if
  `%admin` has a `NOPASSWD` sudo rule then a compromise of this service is a
  direct path to root. To use a dedicated service account:

  ```bash
  sudo dscl . -create /Groups/_parakeet PrimaryGroupID 450
  sudo dscl . -create /Users/_parakeet UniqueID 450
  sudo dscl . -create /Users/_parakeet PrimaryGroupID 450
  sudo dscl . -create /Users/_parakeet UserShell /usr/bin/false
  sudo dscl . -create /Users/_parakeet NFSHomeDirectory /usr/local/var/wyoming-parakeet
  sudo dscl . -create /Users/_parakeet Password '*'
  sudo dscl . -create /Users/_parakeet IsHidden 1
  sudo mkdir -p /usr/local/var/wyoming-parakeet
  sudo chown _parakeet:_parakeet /usr/local/var/wyoming-parakeet

  ./install.sh --user _parakeet
  ```

  Two things to get right. **Keep the checkout outside your home directory** —
  macOS home directories are `drwxr-x---`, so a service account that is not in
  your group cannot traverse into one; `/opt/wyoming-parakeet` works, owned by
  you so `git pull && ./install.sh` still needs no sudo, and read-only to the
  service. And **the model cache follows `HOME`**, which is the service
  account's home, so either let `install.sh` download it as that user or move
  an existing `models--mlx-community--parakeet-tdt-0.6b-v2` directory into
  `<service home>/.cache/huggingface/hub/`.
- **Pin your dependencies** if you care about supply chain. `requirements.txt`
  is deliberately loose so `install.sh` picks up fixes; pin exact versions (and
  ideally hashes) if you would rather audit upgrades. Model weights are
  `safetensors`, so loading them does not execute code, but the initial
  download from HuggingFace is trust-on-first-use — pin a revision if that
  matters to you. `HF_HUB_OFFLINE=1` in the daemon means it never re-fetches
  after that point.

## License

MIT
