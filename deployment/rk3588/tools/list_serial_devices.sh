#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/pydeps${PYTHONPATH:+:$PYTHONPATH}"

echo "Detected serial ports"
echo "---------------------"
python3 - <<'PY'
try:
    from serial.tools import list_ports
except ImportError:
    print("pyserial is not installed")
else:
    for port in sorted(list_ports.comports(), key=lambda item: item.device):
        details = [port.device, port.description or "unknown"]
        if port.vid is not None and port.pid is not None:
            details.append(f"VID:PID={port.vid:04x}:{port.pid:04x}")
        if port.serial_number:
            details.append(f"serial={port.serial_number}")
        print(": ".join(details[:2]), " ".join(details[2:]))
PY

echo
echo "Stable /dev/serial/by-id links"
echo "------------------------------"
if [[ -d /dev/serial/by-id ]]; then
    for link in /dev/serial/by-id/*; do
        [[ -L "$link" ]] || continue
        printf '%s -> %s\n' "${link##*/}" "$(readlink "$link")"
    done | sort
fi
