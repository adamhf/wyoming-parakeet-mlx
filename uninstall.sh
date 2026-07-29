#!/usr/bin/env bash
# Remove the wyoming-parakeet LaunchDaemon. Leaves the checkout and the
# downloaded model in ~/.cache/huggingface alone.
set -euo pipefail

LABEL="local.wyoming-parakeet"
PLIST="/Library/LaunchDaemons/$LABEL.plist"

echo "==> Stopping $LABEL"
sudo launchctl bootout "system/$LABEL" 2>/dev/null || echo "    (not running)"

if [[ -f "$PLIST" ]]; then
    echo "==> Removing $PLIST"
    sudo rm -f "$PLIST"
fi

cat <<MSG

Done. Not removed:
  - this checkout, including .venv
  - the model in ~/.cache/huggingface (delete with:
    rm -rf ~/.cache/huggingface/hub/models--mlx-community--parakeet-tdt-0.6b-v2)
  - the Wyoming integration in Home Assistant (remove it in the UI)
MSG
