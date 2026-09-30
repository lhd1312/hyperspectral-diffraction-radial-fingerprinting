#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-artifacts/yolo13n_pose_lscd_lqe_768x1024_fp32.onnx}"
OUTPUT_DIR="${2:-artifacts}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"
BUILDER_OPT_LEVEL="${BUILDER_OPT_LEVEL:-3}"

mkdir -p "$OUTPUT_DIR"
RAW_ENGINE="$OUTPUT_DIR/yolo13n_pose_lscd_lqe_768x1024_fp16.raw.engine"
ENGINE="$OUTPUT_DIR/yolo13n_pose_lscd_lqe_768x1024_fp16.engine"
TIMING_CACHE="$OUTPUT_DIR/yolo13n_pose_lscd_lqe.timing.cache"
METADATA_JSON="$OUTPUT_DIR/onnx_export_summary.json"
BUILD_LOG="$OUTPUT_DIR/tensorrt_build.log"
BENCHMARK_LOG="$OUTPUT_DIR/tensorrt_benchmark.log"

if command -v trtexec >/dev/null 2>&1; then
  TRTEXEC="$(command -v trtexec)"
elif [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
  TRTEXEC=/usr/src/tensorrt/bin/trtexec
else
  echo "trtexec was not found. Install the JetPack TensorRT development tools." >&2
  exit 2
fi

if [[ ! -f "$ONNX_PATH" ]]; then
  echo "ONNX file not found: $ONNX_PATH" >&2
  exit 2
fi

HELP="$("$TRTEXEC" --help 2>&1 || true)"
BUILD_ARGS=(
  "--onnx=$ONNX_PATH"
  "--saveEngine=$RAW_ENGINE"
  "--fp16"
)
if grep -q -- "--timingCacheFile" <<<"$HELP"; then
  BUILD_ARGS+=("--timingCacheFile=$TIMING_CACHE")
fi
if grep -q -- "--builderOptimizationLevel" <<<"$HELP"; then
  BUILD_ARGS+=("--builderOptimizationLevel=${BUILDER_OPT_LEVEL}")
fi
if grep -q -- "--memPoolSize" <<<"$HELP"; then
  BUILD_ARGS+=("--memPoolSize=workspace:${WORKSPACE_MIB}")
else
  BUILD_ARGS+=("--workspace=${WORKSPACE_MIB}")
fi
if grep -q -- "--skipInference" <<<"$HELP"; then
  BUILD_ARGS+=("--skipInference")
fi

"$TRTEXEC" "${BUILD_ARGS[@]}" 2>&1 | tee "$BUILD_LOG"

"$PYTHON_BIN" wrap_tensorrt_engine.py \
  --onnx "$ONNX_PATH" \
  --metadata-json "$METADATA_JSON" \
  --raw-engine "$RAW_ENGINE" \
  --output-engine "$ENGINE"

BENCH_ARGS=(
  "--loadEngine=$RAW_ENGINE"
  "--warmUp=500"
  "--duration=20"
  "--iterations=200"
)
if grep -q -- "--useCudaGraph" <<<"$HELP"; then
  BENCH_ARGS+=("--useCudaGraph")
fi
"$TRTEXEC" "${BENCH_ARGS[@]}" 2>&1 | tee "$BENCHMARK_LOG"

sha256sum "$ONNX_PATH" "$RAW_ENGINE" "$ENGINE" > "$OUTPUT_DIR/tensorrt_sha256.txt"
echo "[PASS] TensorRT FP16 engine: $ENGINE"
