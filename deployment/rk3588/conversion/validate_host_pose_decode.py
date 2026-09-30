#!/usr/bin/env python3
"""Validate RKNN host decode against exact ONNX and TorchScript predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--exact-raw-onnx", type=Path, required=True)
    parser.add_argument("--stack-onnx", type=Path, required=True)
    parser.add_argument("--torchscript", type=Path, required=True)
    parser.add_argument("--lqe-weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rknn-output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.325)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def load_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("firefly_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def preprocess(path: Path, height: int, width: int) -> tuple[np.ndarray, dict[str, object]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
    metadata = {
        "original_height": source_height,
        "original_width": source_width,
        "input_height": height,
        "input_width": width,
        "scale": scale,
        "pad_left": left,
        "pad_top": top,
    }
    return tensor, metadata


def pack_raw_outputs(outputs: list[np.ndarray]) -> np.ndarray:
    if len(outputs) != 9:
        raise RuntimeError(f"Expected nine raw ONNX tensors, received {len(outputs)}")
    blocks = []
    for scale_index, factor in enumerate((1, 2, 4)):
        tensors = [np.asarray(outputs[scale_index * 3 + offset]).squeeze(0) for offset in range(3)]
        block = np.concatenate(tensors, axis=0)
        if factor > 1:
            block = np.repeat(np.repeat(block, factor, axis=1), factor, axis=2)
        blocks.append(block)
    return np.concatenate(blocks, axis=0)


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "correlation": float(np.corrcoef(reference.reshape(-1), candidate.reshape(-1))[0, 1]),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_absolute_error": float(np.max(np.abs(difference))),
    }


def detections(module, prediction: np.ndarray, preprocess_metadata: dict[str, object], threshold: float):
    found, stats = module.postprocess_decoded_pose(
        prediction,
        preprocess_metadata,
        threshold,
        0.7,
        500,
        100,
        10.0,
    )
    return found, stats


def center_agreement(reference, candidate) -> dict[str, object]:
    reference_xy = np.asarray([(item.x, item.y) for item in reference], dtype=np.float64).reshape(-1, 2)
    candidate_xy = np.asarray([(item.x, item.y) for item in candidate], dtype=np.float64).reshape(-1, 2)
    if not len(reference_xy) or not len(candidate_xy):
        return {"reference_count": len(reference_xy), "candidate_count": len(candidate_xy)}
    distances = np.sqrt(np.sum((reference_xy[:, None] - candidate_xy[None, :]) ** 2, axis=2))
    nearest = np.min(distances, axis=1)
    return {
        "reference_count": int(len(reference_xy)),
        "candidate_count": int(len(candidate_xy)),
        "reference_centers_matched_within_2px": int(np.sum(nearest <= 2.0)),
        "reference_centers_matched_within_5px": int(np.sum(nearest <= 5.0)),
        "mean_nearest_distance_px": float(np.mean(nearest)),
        "max_nearest_distance_px": float(np.max(nearest)),
        "candidate_centers": candidate_xy.astype(int).tolist(),
    }


def main() -> None:
    args = parse_args()
    pipeline = load_pipeline(args.pipeline.resolve())
    images, preprocess_metadata = preprocess(args.image, args.height, args.width)

    torch_model = torch.jit.load(str(args.torchscript), map_location="cpu").eval()
    with torch.inference_mode():
        torch_prediction = torch_model(torch.from_numpy(images)).detach().cpu().numpy().squeeze(0)

    exact_session = ort.InferenceSession(str(args.exact_raw_onnx), providers=["CPUExecutionProvider"])
    exact_outputs = exact_session.run(None, {exact_session.get_inputs()[0].name: images})
    exact_stack = pack_raw_outputs(exact_outputs)
    exact_prediction = pipeline.decode_raw_pose_stack(
        exact_stack.reshape(-1), preprocess_metadata, args.lqe_weights.resolve()
    )

    stack_session = ort.InferenceSession(str(args.stack_onnx), providers=["CPUExecutionProvider"])
    stack_output = stack_session.run(None, {stack_session.get_inputs()[0].name: images})[0].squeeze(0)
    stack_prediction = pipeline.decode_raw_pose_stack(
        stack_output.reshape(-1), preprocess_metadata, args.lqe_weights.resolve()
    )
    rknn_prediction = pipeline.decode_raw_pose_stack(
        np.fromfile(args.rknn_output, dtype=np.float32), preprocess_metadata, args.lqe_weights.resolve()
    )

    torch_detections, torch_stats = detections(pipeline, torch_prediction, preprocess_metadata, args.threshold)
    exact_detections, exact_stats = detections(pipeline, exact_prediction, preprocess_metadata, args.threshold)
    stack_detections, stack_stats = detections(pipeline, stack_prediction, preprocess_metadata, args.threshold)
    rknn_detections, rknn_stats = detections(pipeline, rknn_prediction, preprocess_metadata, args.threshold)
    report = {
        "prediction_shape": list(torch_prediction.shape),
        "decoded_numeric_comparison_to_torchscript": {
            "exact_onnx_plus_numpy_lqe": metrics(torch_prediction, exact_prediction),
            "tanh_gelu_onnx_plus_numpy_lqe": metrics(torch_prediction, stack_prediction),
            "rknn_fp16_plus_numpy_lqe": metrics(torch_prediction, rknn_prediction),
            "rknn_to_tanh_gelu_onnx": metrics(stack_prediction, rknn_prediction),
        },
        "postprocess": {
            "torchscript": {"stats": torch_stats, "centers": [[x.x, x.y] for x in torch_detections]},
            "exact_onnx": {"stats": exact_stats, "agreement": center_agreement(torch_detections, exact_detections)},
            "tanh_gelu_onnx": {"stats": stack_stats, "agreement": center_agreement(torch_detections, stack_detections)},
            "rknn_fp16": {"stats": rknn_stats, "agreement": center_agreement(torch_detections, rknn_detections)},
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
