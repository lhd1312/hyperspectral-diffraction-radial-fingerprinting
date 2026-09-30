#!/usr/bin/env python3
"""Profile a persistent RKNN Lite pose context over a directory of images."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite


INPUT_HEIGHT = 768
INPUT_WIDTH = 1024
RAW_CHANNELS = 204
RAW_HEIGHT = INPUT_HEIGHT // 8
RAW_WIDTH = INPUT_WIDTH // 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def letterbox_rgb(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    original_height, original_width = image_bgr.shape[:2]
    scale = min(INPUT_WIDTH / original_width, INPUT_HEIGHT / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    pad_left = (INPUT_WIDTH - resized_width) // 2
    pad_top = (INPUT_HEIGHT - resized_height) // 2
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((INPUT_HEIGHT, INPUT_WIDTH, 3), 114, dtype=np.uint8)
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    metadata = {
        "original_height": original_height,
        "original_width": original_width,
        "input_height": INPUT_HEIGHT,
        "input_width": INPUT_WIDTH,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
    }
    return np.ascontiguousarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)), metadata


def canonical_raw_stack(output: np.ndarray) -> np.ndarray:
    tensor = np.asarray(output)
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.shape == (RAW_CHANNELS, RAW_HEIGHT, RAW_WIDTH):
        pass
    elif tensor.shape == (RAW_HEIGHT, RAW_WIDTH, RAW_CHANNELS):
        tensor = np.transpose(tensor, (2, 0, 1))
    elif tensor.size == RAW_CHANNELS * RAW_HEIGHT * RAW_WIDTH:
        channel_axis = next((index for index, size in enumerate(tensor.shape) if size == RAW_CHANNELS), None)
        if channel_axis is None:
            raise RuntimeError(f"Cannot identify the 204-channel output layout: {tensor.shape}")
        tensor = np.moveaxis(tensor, channel_axis, 0).reshape(RAW_CHANNELS, RAW_HEIGHT, RAW_WIDTH)
    else:
        raise RuntimeError(f"Unexpected RKNN output shape: {tensor.shape}")
    return np.ascontiguousarray(tensor, dtype=np.float32)


def infer(runtime: RKNNLite, input_rgb: np.ndarray) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    try:
        outputs = runtime.inference(inputs=[input_rgb[np.newaxis, ...]])
    except Exception as first_error:
        try:
            outputs = runtime.inference(inputs=[input_rgb])
        except Exception:
            raise first_error
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not outputs or len(outputs) != 1:
        raise RuntimeError(f"Expected one RKNN output, received {0 if outputs is None else len(outputs)}")
    return canonical_raw_stack(outputs[0]), elapsed_ms


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        path for path in args.images_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No validation images found in {args.images_dir}")
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    runtime = RKNNLite(verbose=False)
    init_started = time.perf_counter()
    result = runtime.load_rknn(str(args.model))
    if result != 0:
        raise RuntimeError(f"load_rknn failed with code {result}")
    core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", None)
    if core_mask is None:
        core_mask = getattr(RKNNLite, "NPU_CORE_ALL", RKNNLite.NPU_CORE_AUTO)
    result = runtime.init_runtime(core_mask=core_mask)
    if result != 0:
        raise RuntimeError(f"init_runtime failed with code {result}")
    model_init_ms = (time.perf_counter() - init_started) * 1000.0

    runs: list[dict[str, object]] = []
    try:
        first_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
        if first_image is None:
            raise FileNotFoundError(image_paths[0])
        warm_input, _ = letterbox_rgb(first_image)
        infer(runtime, warm_input)

        for sample_index, image_path in enumerate(image_paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            input_rgb, preprocess = letterbox_rgb(image)
            output, inference_ms = infer(runtime, input_rgb)
            output_path = (raw_dir / f"sample_{sample_index:03d}_output_0.f32").resolve()
            output.tofile(output_path)
            runs.append(
                {
                    "sample_index": sample_index,
                    "image": str(image_path.resolve()),
                    "preprocess": preprocess,
                    "inference_ms": inference_ms,
                    "outputs": [
                        {
                            "index": 0,
                            "dims": [1, RAW_CHANNELS, RAW_HEIGHT, RAW_WIDTH],
                            "elements": int(output.size),
                            "file": str(output_path),
                        }
                    ],
                }
            )
            print(f"[{sample_index + 1:02d}/{len(image_paths):02d}] {image_path.name}: {inference_ms:.3f} ms", flush=True)
    finally:
        runtime.release()

    manifest = {
        "status": "ok",
        "mode": "batch",
        "backend": "RKNN Lite2 FP16 on RK3588 NPU",
        "model": str(args.model.resolve()),
        "model_init_ms": model_init_ms,
        "warmup_runs": 1,
        "runs": runs,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_init_ms": model_init_ms, "images": len(runs)}, indent=2))


if __name__ == "__main__":
    main()
