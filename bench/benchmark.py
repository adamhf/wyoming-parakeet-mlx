#!/usr/bin/env python3
"""Benchmark speech-to-text backends on the Home Assistant command corpus.

Each backend is loaded once, warmed on every clip, then timed. Reported
latency is the median of the timed pass, which is what a voice assistant
actually experiences -- means are skewed by Metal kernel compilation on the
first request.

Usage:
    ./benchmark.py --backend parakeet:mlx-community/parakeet-tdt-0.6b-v2
    ./benchmark.py --backend mlx-whisper:mlx-community/whisper-large-v3-turbo
    ./benchmark.py --backend faster-whisper:base.en
    ./benchmark.py --backend moonshine:moonshine/base
    ./benchmark.py --backend whispercpp:http://127.0.0.1:8910/inference
"""
import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
CLIPS = BENCH_DIR / "clips"

# Spoken-form numbers a model might emit instead of digits. Home Assistant's
# local intent matching wants digits, so we score this separately.
WORD_NUMBERS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|percent)\b"
)


def normalise(text: str) -> str:
    """Fold away differences that do not change the intent: case, smart
    quotes, punctuation, and whitespace. '%' is kept -- it is semantic."""
    text = unicodedata.normalize("NFKD", text).lower().strip()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^\w\s%']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_corpus():
    rows = []
    for line in (BENCH_DIR / "corpus.tsv").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _tts, expected, has_number = line.split("\t")
        rows.append((expected, has_number == "1"))
    return rows


# --- backends ---------------------------------------------------------------


def read_wav(path):
    """Read a 16kHz mono 16-bit WAV as float32 in [-1, 1).

    Several of these libraries shell out to ffmpeg to load audio, which is an
    unnecessary dependency when the clips are already in the right format --
    and unavailable on the benchmark machine. Feed them arrays instead.
    """
    import wave

    import numpy as np

    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def backend_parakeet(model_id):
    import mlx.core as mx
    from parakeet_mlx import from_pretrained
    from parakeet_mlx.audio import get_logmel

    model = from_pretrained(model_id)

    def transcribe(path):
        mel = get_logmel(mx.array(read_wav(path)), model.preprocessor_config)
        results = model.generate(mel)
        return results[0].text if results else ""

    return transcribe


def backend_mlx_whisper(model_id):
    import mlx_whisper

    def transcribe(path):
        return mlx_whisper.transcribe(
            read_wav(path), path_or_hf_repo=model_id, language="en", fp16=True
        )["text"]

    return transcribe


def backend_faster_whisper(model_id):
    from faster_whisper import WhisperModel

    # Metal is unsupported by CTranslate2; int8 on CPU is the fastest option
    # available on Apple Silicon and is what the HA add-on uses by default.
    model = WhisperModel(model_id, device="cpu", compute_type="int8")

    def transcribe(path):
        segments, _info = model.transcribe(str(path), language="en", beam_size=5)
        return "".join(s.text for s in segments)

    return transcribe


def backend_moonshine(model_id):
    import moonshine_onnx

    model = moonshine_onnx.MoonshineOnnxModel(model_name=model_id)
    tokenizer = moonshine_onnx.load_tokenizer()

    def transcribe(path):
        # Bypass moonshine_onnx.transcribe() so we can supply the audio as an
        # array; it expects shape [batch, samples].
        audio = read_wav(path).reshape(1, -1)
        return " ".join(tokenizer.decode_batch(model.generate(audio)))

    return transcribe


def backend_whispercpp(url):
    import requests

    def transcribe(path):
        with open(path, "rb") as fh:
            r = requests.post(
                url,
                files={"file": fh},
                data={"response_format": "json", "no_context": "true"},
                timeout=120,
            )
        r.raise_for_status()
        return r.json()["text"]

    return transcribe


BACKENDS = {
    "parakeet": backend_parakeet,
    "mlx-whisper": backend_mlx_whisper,
    "faster-whisper": backend_faster_whisper,
    "moonshine": backend_moonshine,
    "whispercpp": backend_whispercpp,
}


# --- runner -----------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, help="kind:model_or_url")
    ap.add_argument("--label", help="Name to report (defaults to --backend)")
    ap.add_argument("--json", action="store_true", help="Emit a JSON result line")
    args = ap.parse_args()

    kind, _, target = args.backend.partition(":")
    if kind not in BACKENDS:
        sys.exit(f"unknown backend {kind!r}; pick one of {', '.join(BACKENDS)}")

    corpus = load_corpus()
    clips = sorted(
        (p for p in CLIPS.glob("*.wav") if p.name != "silence.wav"),
        key=lambda p: (int(p.stem.split("_")[0]), p.stem),
    )
    if not clips:
        sys.exit("no clips found -- run ./make_clips.sh first")

    load_started = time.monotonic()
    transcribe = BACKENDS[kind](target)
    load_seconds = time.monotonic() - load_started

    # Warm every clip first: the first inference compiles kernels, and clip
    # length varies enough that a single warm-up does not cover all shapes.
    for clip in clips:
        transcribe(clip)

    latencies, exact, number_ok, number_total, failures = [], 0, 0, 0, []
    for clip in clips:
        index = int(clip.stem.split("_")[0]) - 1
        expected, has_number = corpus[index]

        started = time.monotonic()
        got = transcribe(clip)
        latencies.append((time.monotonic() - started) * 1000)

        if normalise(got) == normalise(expected):
            exact += 1
        else:
            failures.append((clip.name, normalise(expected), normalise(got)))
        if has_number:
            number_total += 1
            if not WORD_NUMBERS.search(normalise(got)):
                number_ok += 1

    silence = CLIPS / "silence.wav"
    silence_out = normalise(transcribe(silence)) if silence.exists() else "n/a"

    label = args.label or args.backend
    result = {
        "backend": label,
        "clips": len(clips),
        "median_ms": round(statistics.median(latencies), 1),
        "p90_ms": round(sorted(latencies)[int(len(latencies) * 0.9)], 1),
        "exact_match": f"{exact}/{len(clips)}",
        "digits_ok": f"{number_ok}/{number_total}",
        "silence": silence_out,
        "load_s": round(load_seconds, 1),
    }

    if args.json:
        print(json.dumps(result))
    else:
        print(f"\n=== {label} ===")
        for key, value in result.items():
            if key != "backend":
                print(f"  {key:12s} {value}")
        if failures:
            print(f"  mismatches ({len(failures)}):")
            for name, want, got in failures[:8]:
                print(f"    {name}\n      want: {want}\n      got:  {got}")


if __name__ == "__main__":
    main()
