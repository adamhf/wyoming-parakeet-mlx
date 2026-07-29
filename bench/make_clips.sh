#!/usr/bin/env bash
# Render the benchmark corpus to 16kHz mono WAV using macOS TTS, across
# several voices. Synthetic speech is clean and accent-consistent, so treat
# the accuracy numbers as a domain smoke test, not a WER benchmark.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$DIR/clips"
VOICES=(Daniel Samantha Karen)

rm -rf "$OUT"; mkdir -p "$OUT"

i=0
while IFS=$'\t' read -r phrase _expected _has_number; do
    [[ "$phrase" =~ ^#  || -z "$phrase" ]] && continue
    i=$((i + 1))
    for voice in "${VOICES[@]}"; do
        if ! say -v "$voice" -o "$OUT/${i}_${voice}.aiff" "$phrase" 2>/dev/null; then
            say -o "$OUT/${i}_${voice}.aiff" "$phrase"
        fi
        afconvert -f WAVE -d LEI16@16000 -c 1 \
            "$OUT/${i}_${voice}.aiff" "$OUT/${i}_${voice}.wav"
        rm -f "$OUT/${i}_${voice}.aiff"
    done
done < "$DIR/corpus.tsv"

python3 - "$OUT/silence.wav" <<'PY'
import sys, wave
w = wave.open(sys.argv[1], "wb")
w.setparams((1, 2, 16000, 0, "NONE", "NONE"))
w.writeframes(b"\x00\x00" * 16000 * 3)
w.close()
PY

echo "Wrote $(ls "$OUT"/*.wav | wc -l | tr -d ' ') clips to $OUT"
