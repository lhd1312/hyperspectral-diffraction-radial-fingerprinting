#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA_PORT="${1:-}"
if [[ -z "$CAMERA_PORT" ]]; then
  for port in /dev/serial/by-id/*OpenMV* /dev/ttyACM*; do
    if [[ -e "$port" ]]; then
      CAMERA_PORT="$port"
      break
    fi
  done
fi

if [[ -z "$CAMERA_PORT" ]]; then
  echo "[ERROR] OpenMV serial camera was not detected." >&2
  echo "Run: bash tools/list_serial_devices.sh" >&2
  exit 2
fi

export PYTHONPATH="$ROOT/pydeps${PYTHONPATH:+:$PYTHONPATH}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"
export HSI_FULLSCREEN="${HSI_FULLSCREEN:-1}"
if [[ -S /run/user/0/wayland-0 ]]; then
  export XDG_RUNTIME_DIR=/run/user/0
elif [[ -S /run/wayland-0 ]]; then
  export XDG_RUNTIME_DIR=/run
else
  echo "[ERROR] No Weston Wayland socket was found." >&2
  exit 3
fi
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"

echo "[hardware] camera: $CAMERA_PORT"
exec python3 "$ROOT/acquisition/acquire_and_infer.py" \
  --camera-port "$CAMERA_PORT" \
  --preview-only
