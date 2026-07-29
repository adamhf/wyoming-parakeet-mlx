#!/usr/bin/env bash
# Generate the end-to-end test clips using macOS TTS. They are not committed
# because they are trivially reproducible binaries.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/clips"
mkdir -p "$DIR"

PHRASES=(
    "turn off the kitchen lights"
    "set the living room thermostat to twenty one degrees"
    "what is the temperature in the bedroom"
    "dim the hallway lights to thirty percent"
    "is the back door locked"
    "turn on the christmas tree in the conservatory"
    "set a timer for twelve minutes"
    "play radio six music in the kitchen"
    "whats the octopus agile rate right now"
    "close the blinds in the study and turn on the desk lamp"
)

i=1
for phrase in "${PHRASES[@]}"; do
    say -o "$DIR/cmd$i.aiff" "$phrase"
    afconvert -f WAVE -d LEI16@16000 -c 1 "$DIR/cmd$i.aiff" "$DIR/cmd$i.wav"
    rm -f "$DIR/cmd$i.aiff"
    i=$((i + 1))
done

# 3s of digital silence -- a good model returns an empty string here rather
# than hallucinating "Thank you." the way whisper does.
python3 - "$DIR/silence.wav" <<'PY'
import sys, wave
w = wave.open(sys.argv[1], "wb")
w.setparams((1, 2, 16000, 0, "NONE", "NONE"))
w.writeframes(b"\x00\x00" * 16000 * 3)
w.close()
PY

echo "Wrote $(ls "$DIR"/*.wav | wc -l | tr -d ' ') clips to $DIR"
