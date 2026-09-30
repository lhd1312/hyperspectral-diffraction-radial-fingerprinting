#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CAMERA_PORT STAGE_PORT [extra acquisition arguments]" >&2
  echo "Example: $0 /dev/serial/by-id/CAMERA /dev/serial/by-id/STAGE" >&2
  echo "Run tools/list_serial_devices.sh to identify both ports." >&2
  exit 2
fi

CAMERA_PORT="$1"
STAGE_PORT="$2"
shift 2

exec python3 "$ROOT/acquisition/acquire_and_infer.py" \
  --root "$ROOT" \
  --camera-port "$CAMERA_PORT" \
  --stage-port "$STAGE_PORT" \
  --preview \
  --show-scan-window \
  --show-inference-windows \
  --popup-delay-ms 1500 \
  "$@"
