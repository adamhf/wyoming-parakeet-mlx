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

- **An Apple Silicon Mac** (M1 or newer) that stays on. This does not run on
  Intel Macs, a Raspberry Pi, or inside Home Assistant OS.
- **Homebrew Python.** If you are not sure you have it:
  ```bash
  brew install python@3.13
  ```
- **About 3 GB of free disk** — 2.3 GB of model, 600 MB of libraries.

<details>
<summary>Why Homebrew Python specifically</summary>

The installer needs Python 3.10 or newer that includes the `lzma` module.
macOS ships Python 3.9, which is too old, and `pyenv` frequently builds
Python without `lzma` (it needs `xz` present at build time). Both fail — the
pyenv case unhelpfully, passing every version check and then dying at import
time with `ModuleNotFoundError: _lzma`. Homebrew's builds are fine.

`ffmpeg` is **not** required, unlike most speech-to-text tooling. Wyoming
already delivers audio in the right format, so this builds the mel
spectrogram directly.
</details>

## Install

**1. Download and run the installer.**

```bash
git clone https://github.com/adamhf/wyoming-parakeet-mlx
cd wyoming-parakeet-mlx
./install.sh
```

It will ask for your password once, to register the background service.

Expect it to take a few minutes — most of that is downloading the model. It
sets up its own Python environment, runs the test suite, downloads the model,
and registers a background service on port 7892 that starts automatically when
the Mac boots, with no need to log in.

**Keep this folder where it is.** The installer sets things up inside it, so
moving or deleting the folder afterwards breaks the service.

Remove it later with `./uninstall.sh`.

<details>
<summary>Installer options</summary>

`--port`, `--model`, `--user`, `--python`, `--no-daemon`, `--no-download`.
Run `./install.sh --help` for details. See [Security](#security) for why
`--user` is worth using.
</details>

## Connect it to Home Assistant

Installing it is not enough on its own — Home Assistant has to be told about
it, and then told to *use* it. That is two separate steps, and the second one
is easy to miss.

**1. Find your Mac's IP address.** System Settings → Network → click your
active connection → Details → the "IP address" field. Or in a terminal:

```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

That command often lists more addresses than you expect. Ignore anything
belonging to virtualisation software (Parallels, VMware, Docker — commonly
`10.211.55.x`, `10.37.129.x`, `172.17.x.x`) or a VPN (Tailscale uses
`100.x.x.x`); those are not reachable from Home Assistant in the usual case.
You want the one on the same network as Home Assistant itself — if Home
Assistant is on `192.168.1.50`, you want the Mac's `192.168.1.x` address.

If your Mac has both Wi-Fi and Ethernet active you will see two candidates.
Either works, but pick one that will stay put — a wired address, ideally with
a DHCP reservation on your router — because if that interface goes away, Home
Assistant loses speech-to-text.

**2. Add the integration.** In Home Assistant: **Settings → Devices &
Services → Add Integration → Wyoming Protocol**. Enter the IP address from
step 1 and port **7892**.

If it succeeds you get a new entry called `parakeet-mlx`. If it says it cannot
connect, see [Troubleshooting](#troubleshooting).

**3. Actually use it.** Go to **Settings → Voice assistants**, click your
assistant, and set **Speech-to-text** to `parakeet-mlx`. Save.

Until you do this, nothing changes — you will have installed it and Home
Assistant will carry on using whatever it used before. If you have more than
one assistant configured, change each one you care about.

**4. Try it.** Click the Assist icon (top right of Home Assistant), press the
microphone, and say something like "turn off the kitchen lights".

## Check it's working

Is the service running?

```bash
sudo launchctl print system/local.wyoming-parakeet | grep "state ="
```

Expect `state = running`.

Watch it work — leave this running, then speak to Assist:

```bash
tail -f /tmp/local.wyoming-parakeet.stderr
```

Each command produces a line like
`2.70s audio -> 124ms, 45 chars`. The `124ms` is how long transcription took.
The text itself is deliberately not logged; see [Security](#security).

Test it without speaking, using generated audio:

```bash
./test/make-clips.sh
.venv/bin/python test/wy-test.py test/clips/*.wav
```

Expect ten commands transcribed correctly, `silence.wav` coming back empty,
and times around 100–150 ms.

## Troubleshooting

| What you see | What it usually means |
|---|---|
| Home Assistant says **"Failed to connect"** when adding the integration | Wrong IP, or the service is not running. Check `state = running` above, confirm the IP with `ifconfig`, and make sure both machines are on the same network. |
| Integration added fine, but **Assist is no worse and no faster** | You missed step 3 — the assistant is still using its old speech-to-text engine. Settings → Voice assistants → Speech-to-text. |
| **`No matching distribution found for parakeet-mlx`** during install | Wrong Python. Run `brew install python@3.13` and try again, or point the installer at it: `./install.sh --python /opt/homebrew/bin/python3.13`. |
| **`error: service did not start`** at the end of install | Look at `/tmp/local.wyoming-parakeet.stderr` for the reason. If you changed `--user`, re-run `./install.sh` rather than editing the service file by hand. |
| The **first command after a reboot is slow** (~300 ms), then it is fast | Expected. The graphics framework compiles its kernels on first use. It settles by the second or third command. |
| Everything works, then **stops after a few weeks** | Check the Mac has not gone to sleep. System Settings → Displays → Advanced → prevent automatic sleeping. |
| You want to see **what it actually heard** | Add `--debug`, but read [Security](#security) first — the log then contains everything spoken near your microphones. |

## Server options

Most people never need these. `install.sh` bakes them into the service
definition, so to change one, re-run the installer rather than editing the
service file by hand. Run `script/run --help` for the full list.

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
