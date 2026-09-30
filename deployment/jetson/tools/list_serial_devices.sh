#!/usr/bin/env bash
set -euo pipefail

echo "Detected serial ports"
echo "---------------------"
python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/acquisition/acquire_and_infer.py" --list-ports

echo
echo "Stable /dev/serial/by-id links"
echo "------------------------------"
if [[ -d /dev/serial/by-id ]]; then
  find /dev/serial/by-id -maxdepth 1 -type l -printf '%f -> %l\n' | sort
else
  echo "No /dev/serial/by-id directory is present."
fi
