#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-$ROOT/shili}"
OUTPUT_DIR="${2:-$ROOT/results/run_$(date +%Y%m%d_%H%M%S)}"
shift $(( $# >= 2 ? 2 : $# ))

ENGINE="$ROOT/artifacts/yolo13n_pose_lscd_lqe_768x1024_fp16.engine"
CLASSIFIER="$ROOT/models/deployment_classifier_domain_adapted.npz"
WAVELENGTHS="$ROOT/config/selected_wavelengths_real_train_only.csv"
PIPELINE="$ROOT/pipeline/jetson_hsi_end_to_end.py"

for required in "$INPUT_DIR" "$ENGINE" "$CLASSIFIER" "$WAVELENGTHS" "$PIPELINE"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Required input is missing: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
echo "[run] input:  $INPUT_DIR"
echo "[run] output: $OUTPUT_DIR"
echo "[run] engine: $ENGINE"

python3 "$PIPELINE" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --ultralytics-source "$ROOT/../../vendor/ultralytics" \
  --detector-weights "$ENGINE" \
  --classifier-numpy "$CLASSIFIER" \
  --selected-wavelengths "$WAVELENGTHS" \
  --imgsz 768 1024 \
  --device 0 \
  "$@" 2>&1 | tee "$OUTPUT_DIR/run.log"

echo "[PASS] Final overlay: $OUTPUT_DIR/final_result.png"
echo "[PASS] Machine-readable summary: $OUTPUT_DIR/pipeline_summary.json"
