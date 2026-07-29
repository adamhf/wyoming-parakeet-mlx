<!-- Numbers here are reproduced by bench/benchmark.py; see Reproducing below. -->

# Benchmarks

All figures measured on one machine: **M4 Mac mini (10-core, 32 GB), macOS 26.5**.
48 clips — 16 Home Assistant commands rendered through three macOS TTS voices
(Daniel, Samantha, Karen). Every backend loads once, transcribes all 48 clips
to warm up, then runs a timed pass. Latency is the **median** of that pass;
means are skewed by first-request kernel compilation.

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
[Model choice](README.md#model-choice-v2-not-v3) for why that matters more than it looks.
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

## Caveats — read these before trusting the table

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

## Reproducing

**1. Set up a throwaway venv.** Keep the benchmark dependencies out of the
service venv — they pull in CTranslate2, ONNX Runtime and a second copy of
MLX, none of which the server needs:

```bash
python3.13 -m venv /tmp/bench-venv          # explicit version, not bare python3
/tmp/bench-venv/bin/pip install -r bench/requirements.txt
```

Use an explicit interpreter, not bare `python3` — on macOS that is still the
system Python 3.9, which has no MLX wheels and fails with
`No matching distribution found for parakeet-mlx`. Homebrew's
`python3.11`/`3.12`/`3.13`/`3.14` all work.

**2. Render the clips** (macOS only — it uses the built-in `say` voices):

```bash
./bench/make_clips.sh
```

Writes 49 files to `bench/clips/` (16 phrases × 3 voices, plus silence). They
are gitignored; regenerating them is deterministic for a given macOS release,
but voice quality does change between releases, so numbers are only strictly
comparable within one machine.

**3. Run a backend.** The `--backend` argument is `kind:model`:

```bash
/tmp/bench-venv/bin/python bench/benchmark.py \
    --backend parakeet:mlx-community/parakeet-tdt-0.6b-v2
```

Add `--json` for a single machine-readable line instead of the human report,
and `--label` to control the name in the output. Without `--json` it also
prints every mismatch with expected-vs-actual, which is the useful bit when a
model scores worse than you expected.

If you have `HF_HUB_OFFLINE=1` exported (the service sets it), prefix the
command with `HF_HUB_OFFLINE=` so it can fetch models it has not seen.

**The exact commands behind each table row:**

| Row | Command |
|---|---|
| moonshine tiny | `--backend moonshine:moonshine/tiny` |
| moonshine base | `--backend moonshine:moonshine/base` |
| parakeet-tdt-0.6b-v2 | `--backend parakeet:mlx-community/parakeet-tdt-0.6b-v2` |
| parakeet-tdt-0.6b-v3 | `--backend parakeet:mlx-community/parakeet-tdt-0.6b-v3` |
| faster-whisper tiny.en | `--backend faster-whisper:tiny.en` |
| faster-whisper base.en | `--backend faster-whisper:base.en` |
| faster-whisper small.en | `--backend faster-whisper:small.en` |
| faster-whisper distil-large-v3 | `--backend faster-whisper:distil-large-v3` |
| mlx-whisper large-v3-turbo | `--backend mlx-whisper:mlx-community/whisper-large-v3-turbo` |
| mlx-whisper large-v3 | `--backend mlx-whisper:mlx-community/whisper-large-v3-mlx` |
| whisper.cpp \* | `--backend whispercpp:http://127.0.0.1:8920/inference` |

\* The whisper.cpp rows need a server running first. That requires your own
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) build — ideally with
`-DWHISPER_COREML=1` for the CoreML encoder, which is what the table
measured:

```bash
./build/bin/whisper-server -m models/ggml-large-v3-turbo.bin \
    --host 127.0.0.1 --port 8920 -nc -sns -l en -t 4
```

`-nc` matters. Without it the server carries decoder context between requests
and periodically returns the previous utterance, which would make whisper.cpp
look far worse than it is. The table gives it its best configuration
deliberately.

**Cost.** The full sweep is roughly 15 minutes of compute, dominated by
`distil-large-v3` at ~4.3 s per clip, plus however long it takes to pull about
7 GB of models. To clean up afterwards:

```bash
rm -rf /tmp/bench-venv bench/clips
rm -rf ~/.cache/huggingface/hub/models--mlx-community--whisper-large-v3*
rm -rf ~/.cache/huggingface/hub/models--Systran--faster-*
rm -rf ~/.cache/huggingface/hub/models--UsefulSensors--moonshine*
```
