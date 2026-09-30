#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:-$ROOT/shili}"
OUTPUT_DIR="${2:-$ROOT/results/run_$(date +%Y%m%d_%H%M%S)}"
shift $(( $# >= 2 ? 2 : $# ))

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
PIPELINE="$ROOT/pipeline/rk3588_hsi_end_to_end.py"
CLASSIFIER="$ROOT/models/deployment_classifier_domain_adapted.npz"
WAVELENGTHS="$ROOT/config/selected_wavelengths_real_train_only.csv"
RKNN_MODEL="$ROOT/models/yolo13n_pose_lscd_stack_tanhgelu_768x1024_rk3588_fp16.rknn"
RKNN_RUNNER="$ROOT/tools/rknn_lite_pose_runner.py"
LQE_WEIGHTS="$ROOT/models/yolo13n_pose_lqe_weights.npz"

for required in "$INPUT_DIR" "$PIPELINE" "$CLASSIFIER" "$WAVELENGTHS" \
  "$RKNN_MODEL" "$RKNN_RUNNER" "$LQE_WEIGHTS" "$PYTHON_BIN"; do
  if [[ ! -e "$required" ]]; then
    echo "[ERROR] Required input is missing: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$ROOT/pydeps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
if [[ "${QT_QPA_PLATFORM:-}" == "wayland" && -S /run/wayland-0 ]]; then
  HSI_RUNTIME_DIR=/run/hsi-wayland-root
  mkdir -p "$HSI_RUNTIME_DIR"
  chmod 700 "$HSI_RUNTIME_DIR"
  ln -sfn ../wayland-0 "$HSI_RUNTIME_DIR/wayland-0"
  export XDG_RUNTIME_DIR="$HSI_RUNTIME_DIR"
  export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
fi

echo "[run] input:    $INPUT_DIR"
echo "[run] output:   $OUTPUT_DIR"
echo "[run] backend:  RK3588 RKNN NPU"
echo "[run] detector: $RKNN_MODEL"

"$PYTHON_BIN" "$PIPELINE" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --detector-weights "$RKNN_MODEL" \
  --detector-backend rknn \
  --rknn-runner "$RKNN_RUNNER" \
  --lqe-weights "$LQE_WEIGHTS" \
  --classifier-numpy "$CLASSIFIER" \
  --selected-wavelengths "$WAVELENGTHS" \
  --imgsz 768 1024 \
  "$@" 2>&1 | tee "$OUTPUT_DIR/run.log"

echo "[PASS] Final overlay: $OUTPUT_DIR/final_result.png"
echo "[PASS] Summary:       $OUTPUT_DIR/pipeline_summary.json"
