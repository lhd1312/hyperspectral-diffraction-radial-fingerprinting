#!/usr/bin/env python3
"""Evaluate batch RKNN pose outputs against YOLO-Pose center labels."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve a dense rectangular assignment problem without SciPy."""
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost must be a two-dimensional matrix")
    row_count, column_count = matrix.shape
    transposed = row_count > column_count
    if transposed:
        matrix = matrix.T
        row_count, column_count = matrix.shape

    u = np.zeros(row_count + 1, dtype=np.float64)
    v = np.zeros(column_count + 1, dtype=np.float64)
    p = np.zeros(column_count + 1, dtype=np.int64)
    way = np.zeros(column_count + 1, dtype=np.int64)
    for row in range(1, row_count + 1):
        p[0] = row
        minimum = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = p[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = matrix[current_row - 1, candidate - 1] - u[current_row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    way[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break

    rows = []
    columns = []
    for column in range(1, column_count + 1):
        if p[column] != 0:
            rows.append(p[column] - 1)
            columns.append(column - 1)
    row_indices = np.asarray(rows, dtype=np.int64)
    column_indices = np.asarray(columns, dtype=np.int64)
    if transposed:
        return column_indices, row_indices
    return row_indices, column_indices


def match_with_radius(distance: np.ndarray, radius: float) -> list[tuple[int, int, float]]:
    """Radius-aware assignment with the study's finite unmatched penalty.

    This preserves the archived evaluation, not a lexicographic maximum-cardinality rule.
    """
    reference_count, prediction_count = distance.shape
    size = reference_count + prediction_count
    unmatched_cost = radius + 1.0
    invalid_cost = unmatched_cost * (size + 1)
    cost = np.zeros((size, size), dtype=np.float64)
    cost[:reference_count, :prediction_count] = np.where(
        distance <= radius, distance, invalid_cost
    )
    cost[:reference_count, prediction_count:] = unmatched_cost
    cost[reference_count:, :prediction_count] = unmatched_cost
    reference_indices, prediction_indices = linear_sum_assignment(cost)
    pairs = []
    for reference_index, prediction_index in zip(reference_indices, prediction_indices):
        if reference_index >= reference_count or prediction_index >= prediction_count:
            continue
        error = float(distance[reference_index, prediction_index])
        if error <= radius:
            pairs.append((int(reference_index), int(prediction_index), error))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--lqe-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.325)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--match-radius", type=float, default=10.0)
    parser.add_argument("--crop-size", type=int, default=100)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--backend-name", default="RKNN FP16 on RK3588 NPU")
    return parser.parse_args()


def load_pipeline(path: Path):
    spec = importlib.util.spec_from_file_location("firefly_pipeline", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def filter_edge_centers(centers: np.ndarray, width: int, height: int, crop_size: int) -> np.ndarray:
    margin = crop_size / 2.0
    keep = (
        (centers[:, 0] >= margin)
        & (centers[:, 0] <= width - margin)
        & (centers[:, 1] >= margin)
        & (centers[:, 1] <= height - margin)
    )
    return centers[keep]


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    pipeline = load_pipeline(args.pipeline.resolve())
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("mode") != "batch" or not manifest.get("runs"):
        raise ValueError("Expected a non-empty RKNN batch manifest")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    per_image: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    total_tp = total_fp = total_fn = total_gt = total_pred = 0
    count_absolute_errors: list[float] = []
    inference_ms: list[float] = []

    for run in manifest["runs"]:
        image_path = Path(run["image"])
        if args.images_dir is not None:
            image_path = args.images_dir / image_path.name
        preprocess = run["preprocess"]
        width = int(preprocess["original_width"])
        height = int(preprocess["original_height"])
        output_path = Path(run["outputs"][0]["file"])
        if args.raw_dir is not None:
            output_path = args.raw_dir / output_path.name
        raw = np.fromfile(output_path, dtype=np.float32)
        prediction = pipeline.decode_raw_pose_stack(raw, preprocess, args.lqe_weights.resolve())
        detections, stats = pipeline.postprocess_decoded_pose(
            prediction,
            preprocess,
            args.threshold,
            args.iou,
            500,
            args.crop_size,
            10.0,
        )

        label_path = args.labels / f"{image_path.stem}.txt"
        references = pipeline.read_reference_centers(label_path, width, height)
        references = filter_edge_centers(references, width, height, args.crop_size)
        predicted = np.asarray([(item.x, item.y) for item in detections], dtype=np.float64).reshape(-1, 2)
        matched = 0
        valid_pairs: list[tuple[int, int, float]] = []
        if len(references) and len(predicted):
            distance = np.linalg.norm(references[:, None, :] - predicted[None, :, :], axis=2)
            valid_pairs = match_with_radius(distance, args.match_radius)
            matched = len(valid_pairs)
        false_positive = len(detections) - matched
        false_negative = len(references) - matched
        total_tp += matched
        total_fp += false_positive
        total_fn += false_negative
        total_gt += len(references)
        total_pred += len(detections)
        count_error = len(detections) - len(references)
        count_absolute_errors.append(abs(count_error))
        inference_ms.append(float(run["inference_ms"]))

        precision = safe_ratio(matched, len(detections))
        recall = safe_ratio(matched, len(references))
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_image.append(
            {
                "image": image_path.name,
                "ground_truth_centers": len(references),
                "predicted_centers": len(detections),
                "true_positive": matched,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "precision": precision,
                "recall": recall,
                "f1_at_10px": f1,
                "count_error": count_error,
                "absolute_count_error": abs(count_error),
                "inference_ms": float(run["inference_ms"]),
                "raw_candidates": stats["raw_predictions"],
            }
        )
        for detection_index, item in enumerate(detections, start=1):
            prediction_rows.append(
                {
                    "image": image_path.name,
                    "prediction_index": detection_index,
                    "x_px": item.x,
                    "y_px": item.y,
                    "box_confidence": item.box_confidence,
                    "keypoint_confidence": item.keypoint_confidence,
                }
            )
        for reference_index, prediction_index, error in valid_pairs:
            matched_rows.append(
                {
                    "image": image_path.name,
                    "reference_index": reference_index + 1,
                    "prediction_index": prediction_index + 1,
                    "reference_x_px": float(references[reference_index, 0]),
                    "reference_y_px": float(references[reference_index, 1]),
                    "prediction_x_px": float(predicted[prediction_index, 0]),
                    "prediction_y_px": float(predicted[prediction_index, 1]),
                    "center_error_px": error,
                }
            )

    precision = safe_ratio(total_tp, total_tp + total_fp)
    recall = safe_ratio(total_tp, total_tp + total_fn)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = np.asarray([row["center_error_px"] for row in matched_rows], dtype=np.float64)
    inference = np.asarray(inference_ms, dtype=np.float64)
    summary = {
        "backend": args.backend_name,
        "images": len(per_image),
        "match_radius_px": args.match_radius,
        "confidence_threshold": args.threshold,
        "ground_truth_centers": total_gt,
        "predicted_centers": total_pred,
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "precision": precision,
        "recall": recall,
        "f1_at_10px": f1,
        "count_mae": float(np.mean(count_absolute_errors)),
        "count_rmse": float(np.sqrt(np.mean(np.square(count_absolute_errors)))),
        "center_error_mean_px": float(np.mean(errors)),
        "center_error_median_px": float(np.median(errors)),
        "center_error_p95_px": float(np.percentile(errors, 95)),
        "center_error_max_px": float(np.max(errors)),
        "model_init_ms": float(manifest["model_init_ms"]),
        "steady_state_inference_ms": {
            "mean": float(np.mean(inference)),
            "std": float(np.std(inference, ddof=1)),
            "min": float(np.min(inference)),
            "max": float(np.max(inference)),
            "fps": float(1000.0 / np.mean(inference)),
        },
    }
    (args.output_dir / "rknn_pose_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "per_image_metrics.csv", per_image, list(per_image[0]))
    write_csv(args.output_dir / "matched_center_errors.csv", matched_rows, list(matched_rows[0]))
    write_csv(args.output_dir / "predicted_centers.csv", prediction_rows, list(prediction_rows[0]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
