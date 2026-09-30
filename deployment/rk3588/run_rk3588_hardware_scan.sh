#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA_PORT=""
STAGE_PORT=""
if [[ $# -ge 2 && "$1" != --* && "$2" != --* ]]; then
  CAMERA_PORT="$1"
  STAGE_PORT="$2"
  shift 2
else
  for port in /dev/serial/by-id/*OpenMV* /dev/ttyACM*; do
    if [[ -e "$port" ]]; then
      CAMERA_PORT="$port"
      break
    fi
  done
  for port in /dev/serial/by-id/*USB2.0-Serial* /dev/serial/by-id/*1a86* /dev/ttyUSB*; do
    if [[ -e "$port" ]]; then
      STAGE_PORT="$port"
      break
    fi
  done
fi

if [[ -z "$CAMERA_PORT" ]]; then
  echo "[ERROR] OpenMV serial camera was not detected." >&2
  echo "Connect OpenMV, ensure main.py starts automatically, then run tools/list_serial_devices.sh." >&2
  exit 2
fi
if [[ -z "$STAGE_PORT" ]]; then
  echo "[ERROR] Translation-stage serial port was not detected." >&2
  echo "OpenMV is available at: $CAMERA_PORT" >&2
  if lsusb 2>/dev/null | grep -qi '1a86:7523'; then
    echo "The CH341 USB adapter is visible, but this kernel has not created /dev/ttyUSB0." >&2
  else
    echo "Connect and power the stage USB serial adapter, then run tools/list_serial_devices.sh." >&2
  fi
  exit 2
fi

echo "[hardware] camera: $CAMERA_PORT"
echo "[hardware] stage:  $STAGE_PORT"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
export PYTHONPATH="$ROOT/pydeps${PYTHONPATH:+:$PYTHONPATH}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"
export HSI_FULLSCREEN="${HSI_FULLSCREEN:-1}"
if [[ -S /run/user/0/wayland-0 ]]; then
  export XDG_RUNTIME_DIR=/run/user/0
elif [[ -S /run/wayland-0 ]]; then
  HSI_RUNTIME_DIR=/run/hsi-wayland-root
  mkdir -p "$HSI_RUNTIME_DIR"
  chmod 700 "$HSI_RUNTIME_DIR"
  ln -sfn ../wayland-0 "$HSI_RUNTIME_DIR/wayland-0"
  export XDG_RUNTIME_DIR="$HSI_RUNTIME_DIR"
else
  echo "No Weston Wayland socket was found; start the local HDMI desktop first." >&2
  exit 3
fi
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"

DEPLOYMENT_ARGS=()
if [[ "${HSI_SAVE_CUBES:-0}" != "1" ]]; then
  DEPLOYMENT_ARGS+=(--no-save-cubes)
fi
if [[ "${HSI_FULL_OUTPUT:-0}" != "1" ]]; then
  DEPLOYMENT_ARGS+=(--minimal-output)
fi
if [[ "${HSI_ASSUME_YES:-0}" == "1" ]]; then
  DEPLOYMENT_ARGS+=(--assume-yes)
fi

exec "$PYTHON_BIN" "$ROOT/acquisition/acquire_and_infer.py" \
  --root "$ROOT" \
  --camera-port "$CAMERA_PORT" \
  --stage-port "$STAGE_PORT" \
  --preview \
  --show-scan-window \
  --show-inference-windows \
  --popup-delay-ms "${HSI_POPUP_DELAY_MS:-3000}" \
  "${DEPLOYMENT_ARGS[@]}" \
  "$@"
