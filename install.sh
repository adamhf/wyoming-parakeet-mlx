#!/usr/bin/env bash
# Install wyoming-parakeet and (optionally) register it as a LaunchDaemon.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_DIR/.venv"
LABEL="local.wyoming-parakeet"
PLIST="/Library/LaunchDaemons/$LABEL.plist"

PORT=7892
MODEL="mlx-community/parakeet-tdt-0.6b-v2"
SERVICE_USER="$(id -un)"
PYTHON=""
INSTALL_DAEMON=1
DOWNLOAD_MODEL=1

usage() {
    cat <<EOF
Usage: ./install.sh [options]

  --port PORT       Wyoming port (default: $PORT)
  --model ID        HuggingFace model id (default: $MODEL)
  --user USER       User the daemon runs as (default: $SERVICE_USER)
  --python PATH     Python interpreter to build the venv from
  --no-daemon       Set up the venv only; don't install the LaunchDaemon
  --no-download     Skip pre-downloading the model
  -h, --help        Show this help

Installs in place, so keep the checkout somewhere permanent.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --user) SERVICE_USER="$2"; shift 2 ;;
        --python) PYTHON="$2"; shift 2 ;;
        --no-daemon) INSTALL_DAEMON=0; shift ;;
        --no-download) DOWNLOAD_MODEL=0; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

die() { echo "error: $*" >&2; exit 1; }

# --- preflight -------------------------------------------------------------

[[ "$(uname -s)" == "Darwin" ]] || die "macOS only (MLX is Apple Silicon)."
[[ "$(uname -m)" == "arm64" ]] || die "Apple Silicon required; this is $(uname -m)."
id -u "$SERVICE_USER" >/dev/null 2>&1 || die "no such user: $SERVICE_USER"
# Run as the service account's own primary group. Hardcoding "staff" would
# hand a dedicated service user read access to every staff-group path,
# including other users' home directories -- which defeats the isolation.
SERVICE_GROUP="$(id -gn "$SERVICE_USER")"

# librosa pulls in pooch, which imports lzma. Pythons built without xz (a
# common pyenv default) satisfy every version check and then fail at import
# time with ModuleNotFoundError: _lzma -- so check for it up front.
usable_python() {
    local py resolved
    py="$1"
    resolved="$(command -v "$py" 2>/dev/null)" || return 1
    [[ -x "$resolved" ]] || return 1
    "$resolved" -c 'import sys, lzma; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        >/dev/null 2>&1 || return 1
    echo "$resolved"
}

if [[ -n "$PYTHON" ]]; then
    PYTHON="$(usable_python "$PYTHON")" \
        || die "$PYTHON is unusable: needs >=3.10 and the lzma module."
else
    # Search PATH *and* the usual Homebrew prefixes explicitly. A
    # non-interactive shell often has neither on PATH, which would otherwise
    # leave us falling back to the system python3 (3.9, too old).
    for candidate in \
        python3.14 python3.13 python3.12 python3.11 python3 \
        /opt/homebrew/bin/python3.1{4,3,2,1} /opt/homebrew/bin/python3 \
        /usr/local/bin/python3.1{4,3,2,1} /usr/local/bin/python3
    do
        if PYTHON="$(usable_python "$candidate")"; then
            break
        fi
        PYTHON=""
    done
    [[ -n "$PYTHON" ]] || die "no suitable python found (needs >=3.10 with the lzma module; try: brew install python@3.13)"
fi

echo "==> Python:  $PYTHON ($("$PYTHON" -V 2>&1))"
echo "==> Install: $REPO_DIR"
echo "==> Model:   $MODEL"
echo "==> Port:    $PORT"
echo "==> User:    $SERVICE_USER:$SERVICE_GROUP"

# --- venv ------------------------------------------------------------------

echo "==> Creating virtualenv"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
echo "==> Installing dependencies (this pulls ~600MB of MLX wheels)"
"$VENV/bin/pip" install --quiet -r "$REPO_DIR/requirements.txt"

echo "==> Running unit tests"
"$VENV/bin/pip" install --quiet pytest pytest-asyncio
( cd "$REPO_DIR" && "$VENV/bin/python" -m pytest -q )

if [[ "$DOWNLOAD_MODEL" -eq 1 ]]; then
    echo "==> Downloading $MODEL (~2.3GB on first run)"
    HF_HUB_OFFLINE= "$VENV/bin/python" - "$MODEL" <<'EOF'
import sys
from parakeet_mlx import from_pretrained
from_pretrained(sys.argv[1])
EOF
fi

if [[ "$INSTALL_DAEMON" -eq 0 ]]; then
    echo
    echo "Done (no daemon installed). Run it with:"
    echo "  $REPO_DIR/script/run --uri tcp://0.0.0.0:$PORT --model $MODEL"
    exit 0
fi

# --- launchd ---------------------------------------------------------------

echo "==> Installing LaunchDaemon (needs sudo)"
TMP_PLIST="$(mktemp -t wyoming-parakeet)"
trap 'rm -f "$TMP_PLIST"' EXIT
cat > "$TMP_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>Label</key>
	<string>$LABEL</string>
	<key>ProgramArguments</key>
	<array>
		<string>$REPO_DIR/script/run</string>
		<string>--uri</string>
		<string>tcp://0.0.0.0:$PORT</string>
		<string>--model</string>
		<string>$MODEL</string>
	</array>
	<key>EnvironmentVariables</key>
	<dict>
		<key>HF_HUB_OFFLINE</key><string>1</string>
		<key>HOME</key><string>$(eval echo "~$SERVICE_USER")</string>
	</dict>
	<key>UserName</key><string>$SERVICE_USER</string>
	<key>GroupName</key><string>$SERVICE_GROUP</string>
	<key>InitGroups</key><true/>
	<key>WorkingDirectory</key><string>$REPO_DIR</string>
	<key>RunAtLoad</key><true/>
	<key>KeepAlive</key><true/>
	<key>ProcessType</key><string>Interactive</string>
	<key>LowPriorityIO</key><false/>
	<key>StandardOutPath</key><string>/tmp/$LABEL.stdout</string>
	<key>StandardErrorPath</key><string>/tmp/$LABEL.stderr</string>
</dict>
</plist>
EOF

plutil -lint "$TMP_PLIST" >/dev/null || die "generated plist is malformed"
sudo cp "$TMP_PLIST" "$PLIST"
sudo chown root:wheel "$PLIST"
sudo chmod 644 "$PLIST"

# bootout returns before the job is actually gone; bootstrapping into the
# gap fails with "Bootstrap failed: 5: Input/output error" and leaves nothing
# loaded at all. Wait for it to disappear first.
sudo launchctl bootout "system/$LABEL" 2>/dev/null || true
for _ in $(seq 1 30); do
    sudo launchctl print "system/$LABEL" >/dev/null 2>&1 || break
    sleep 1
done
if sudo launchctl print "system/$LABEL" >/dev/null 2>&1; then
    die "$LABEL is still loaded after bootout; unload it manually and retry"
fi

sudo launchctl bootstrap system "$PLIST" \
    || die "bootstrap failed; the service is not running. Check: sudo launchctl print system/$LABEL"

echo "==> Waiting for the service to come up"
for _ in $(seq 1 60); do
    if nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
        echo "    listening on $PORT"
        break
    fi
    sleep 2
done

nc -z 127.0.0.1 "$PORT" 2>/dev/null || die "service did not start; check /tmp/$LABEL.stderr"

cat <<EOF

Done. Add it in Home Assistant under Settings -> Devices & Services ->
Add Integration -> Wyoming Protocol, using this host and port $PORT,
then select the new engine as the speech-to-text step in your Assist pipeline.

  logs:      tail -f /tmp/$LABEL.stderr
  verify:    $VENV/bin/python test/wy-test.py test/clips/*.wav
             (run test/make-clips.sh first to generate the clips)
EOF
