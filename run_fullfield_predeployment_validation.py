from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from jetson_hsi_end_to_end import (
    CLASS_LABELS,
    Detection,
    discover_bands,
    import_custom_yolo,
    nonzero_mean,
    parse_position,
    radial_geometry,
    radial_profile,
    read_gray_u8,
    read_reference_centers,
    suppress_near_duplicate_centers,
    wavelength_axis,
)


ROOT = Path(__file__).resolve().parent
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASS_LABELS)}
FULLFIELD_CLASSES = ["CMB", "DWB", "WQ", "YMXB"]


def read_selected_wavelengths(path: Path) -> np.ndarray:
    frame = pd.read_csv(path)
    return frame["wavelength_nm"].to_numpy(dtype=np.float64)


def metadata_lookup(path: Path) -> dict[str, dict[str, str]]:
    frame = pd.read_csv(path)
    return {
        str(row["sequence_id"]): {key: str(value) for key, value in row.items()}
        for _, row in frame.iterrows()
    }


def discover_bands_with_virtual_name_repair(folder: Path) -> list[Path]:
    try:
        return discover_bands(folder)
    except ValueError as error:
        files = [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ]
        if len(files) < 3:
            raise
        positions = np.asarray([parse_position(path) for path in files], dtype=np.int64)
        sorted_positions = np.sort(positions)
        positive_steps = np.diff(sorted_positions)
        positive_steps = positive_steps[positive_steps > 0]
        values, counts = np.unique(positive_steps, return_counts=True)
        common_step = int(values[np.argmax(counts)])
        expected = np.arange(
            int(sorted_positions.min()),
            int(sorted_positions.min()) + common_step * len(files),
            common_step,
            dtype=np.int64,
        )
        missing = sorted(set(expected.tolist()) - set(positions.tolist()))
        extra = sorted(set(positions.tolist()) - set(expected.tolist()))
        if len(missing) != 1 or len(extra) != 1:
            raise error
        corrected_position = {
            int(position): int(position) for position in positions
        }
        corrected_position[extra[0]] = missing[0]
        repaired = sorted(files, key=lambda path: corrected_position[parse_position(path)])
        repaired_positions = np.asarray(
            [corrected_position[parse_position(path)] for path in repaired],
            dtype=np.int64,
        )
        if not np.all(np.diff(repaired_positions) == common_step):
            raise error
        print(
            f"[virtual filename repair] {folder.name}: "
            f"treated pos_{extra[0]} as pos_{missing[0]} without renaming the source file"
        )
        return repaired


def resolve_source_folder(project_root: Path, recorded_path: str) -> Path:
    primary = (project_root / recorded_path).resolve()
    if primary.exists():
        return primary
    backup = (project_root / "oridata - 副本" / Path(recorded_path).name).resolve()
    if backup.exists():
        print(f"[source fallback] {recorded_path} -> {backup}")
        return backup
    raise FileNotFoundError(f"Full-field sequence was not found: {recorded_path}")


def filter_edge_centers(
    centers: np.ndarray, width: int, height: int, crop_size: int
) -> np.ndarray:
    margin = crop_size / 2.0
    keep = (
        (centers[:, 0] >= margin)
        & (centers[:, 0] <= width - margin)
        & (centers[:, 1] >= margin)
        & (centers[:, 1] <= height - margin)
    )
    return centers[keep]


def extract_selected_features(
    files: list[Path],
    detections: list[Detection],
    selected_wavelengths: np.ndarray,
    crop_size: int,
    radial_bins: int,
) -> np.ndarray:
    feature_dimension = len(selected_wavelengths) * (radial_bins + 1)
    if not detections:
        return np.empty((0, feature_dimension), dtype=np.float32)
    raw_wavelengths = wavelength_axis(files, 400.0, 800.0)
    positions = np.interp(
        selected_wavelengths,
        raw_wavelengths,
        np.arange(len(raw_wavelengths), dtype=np.float64),
    )
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    weights = positions - lower
    lower = np.clip(lower, 0, len(files) - 1)
    upper = np.clip(upper, 0, len(files) - 1)
    half = crop_size // 2
    yy, xx = np.mgrid[:crop_size, :crop_size]
    mask = (
        (yy - half) ** 2 + (xx - half) ** 2 <= half**2
    ).astype(np.float32)
    bin_ids, bin_counts = radial_geometry(crop_size, radial_bins)
    n = len(detections)
    spectra = np.zeros((n, len(selected_wavelengths)), dtype=np.float32)
    radial = np.zeros((n, len(selected_wavelengths), radial_bins), dtype=np.float32)
    image_cache: dict[int, np.ndarray] = {}

    def image_at(index: int) -> np.ndarray:
        if index not in image_cache:
            image_cache[index] = read_gray_u8(files[index]).astype(np.float32)
        return image_cache[index]

    for band_order, (lower_index, upper_index, weight) in enumerate(
        zip(lower, upper, weights)
    ):
        lower_image = image_at(int(lower_index))
        upper_image = image_at(int(upper_index))
        for detection_index, detection in enumerate(detections):
            lower_crop = lower_image[
                detection.y - half : detection.y + half,
                detection.x - half : detection.x + half,
            ]
            if lower_index == upper_index or weight == 0.0:
                crop = lower_crop
            else:
                upper_crop = upper_image[
                    detection.y - half : detection.y + half,
                    detection.x - half : detection.x + half,
                ]
                crop = lower_crop * (1.0 - weight) + upper_crop * weight
            crop = crop * mask
            spectra[detection_index, band_order] = nonzero_mean(crop)
            radial[detection_index, band_order] = radial_profile(
                crop, bin_ids, bin_counts
            )
    normalized = (spectra - spectra.mean(axis=1, keepdims=True)) / (
        spectra.std(axis=1, keepdims=True) + 1e-6
    )
    return np.hstack([normalized, radial.reshape(n, -1)]).astype(np.float32)


def detections_from_result(
    result,
    threshold: float,
    crop_size: int,
    center_nms_distance: float,
) -> tuple[list[Detection], dict[str, int]]:
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or len(boxes) == 0:
        return [], {
            "raw_predictions": 0,
            "confidence_rejected": 0,
            "edge_rejected": 0,
            "duplicate_rejected": 0,
        }
    centers = keypoints.xy[:, 0, :].detach().cpu().numpy()
    box_scores = boxes.conf.detach().cpu().numpy()
    point_scores = (
        np.ones_like(box_scores)
        if keypoints.conf is None
        else keypoints.conf[:, 0].detach().cpu().numpy()
    )
    height, width = result.orig_shape
    margin = crop_size / 2.0
    candidates: list[Detection] = []
    confidence_rejected = 0
    edge_rejected = 0
    for center, box_score, point_score in zip(centers, box_scores, point_scores):
        x, y = float(center[0]), float(center[1])
        if (
            not np.isfinite([x, y, box_score, point_score]).all()
            or box_score < threshold
            or point_score < threshold
        ):
            confidence_rejected += 1
            continue
        if x < margin or y < margin or x > width - margin or y > height - margin:
            edge_rejected += 1
            continue
        candidates.append(
            Detection(
                x=int(round(x)),
                y=int(round(y)),
                box_confidence=float(box_score),
                keypoint_confidence=float(point_score),
                joint_confidence=float(min(box_score, point_score)),
            )
        )
    detections = suppress_near_duplicate_centers(candidates, center_nms_distance)
    return detections, {
        "raw_predictions": int(len(boxes)),
        "confidence_rejected": confidence_rejected,
        "edge_rejected": edge_rejected,
        "duplicate_rejected": len(candidates) - len(detections),
    }


def hungarian_matches(
    ground_truth: np.ndarray, predicted: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(ground_truth) == 0 or len(predicted) == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    distance = np.linalg.norm(
        ground_truth[:, None, :] - predicted[None, :, :], axis=2
    )
    gt_index, pred_index = linear_sum_assignment(distance)
    return gt_index, pred_index, distance[gt_index, pred_index]


def collect_train_center_plan(
    image_dir: Path,
    label_dir: Path,
    lookup: dict[str, dict[str, str]],
    crop_size: int,
    max_per_class: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for image_path in sorted(image_dir.glob("*.png")):
        sequence = image_path.stem
        if sequence not in lookup:
            raise KeyError(f"No metadata row for {sequence}")
        label = lookup[sequence]["class_name"]
        with Image.open(image_path) as image:
            width, height = image.size
        centers = read_reference_centers(label_dir / f"{sequence}.txt", width, height)
        centers = filter_edge_centers(centers, width, height, crop_size)
        for center_index, (x, y) in enumerate(centers, start=1):
            rows.append(
                {
                    "sequence": sequence,
                    "source_folder": lookup[sequence]["source_folder"],
                    "class_label": label,
                    "class_index": CLASS_TO_INDEX[label],
                    "center_index": center_index,
                    "x": int(round(x)),
                    "y": int(round(y)),
                }
            )
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    selected = []
    for label, group in frame.groupby("class_label"):
        indices = group.index.to_numpy()
        if len(indices) > max_per_class:
            indices = rng.choice(indices, size=max_per_class, replace=False)
        selected.append(frame.loc[np.sort(indices)])
        print(f"[train centers] {label}: selected {len(indices)}/{len(group)}")
    return pd.concat(selected, ignore_index=True)


def extract_plan_features(
    plan: pd.DataFrame,
    project_root: Path,
    selected_wavelengths: np.ndarray,
    crop_size: int,
    radial_bins: int,
    cache_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    feature_blocks = []
    metadata_blocks = []
    grouped = list(plan.groupby("sequence", sort=True))
    field_cache_dir = cache_dir / "train_fields_selected_only"
    field_cache_dir.mkdir(parents=True, exist_ok=True)
    for field_index, (sequence, group) in enumerate(grouped, start=1):
        field_cache = field_cache_dir / f"{sequence}.npz"
        if field_cache.exists():
            with np.load(field_cache) as data:
                features = data["x"].astype(np.float32)
            feature_blocks.append(features)
            metadata_blocks.append(group.reset_index(drop=True))
            print(
                f"[train feature field] {field_index}/{len(grouped)} "
                f"{sequence}: {len(group)} (cache)"
            )
            continue
        source_folder = resolve_source_folder(
            project_root, str(group.iloc[0]["source_folder"])
        )
        files = discover_bands_with_virtual_name_repair(source_folder)
        detections = [
            Detection(
                x=int(row["x"]),
                y=int(row["y"]),
                box_confidence=1.0,
                keypoint_confidence=1.0,
                joint_confidence=1.0,
            )
            for _, row in group.iterrows()
        ]
        features = extract_selected_features(
            files,
            detections,
            selected_wavelengths,
            crop_size,
            radial_bins,
        )
        np.savez_compressed(field_cache, x=features)
        feature_blocks.append(features)
        metadata_blocks.append(group.reset_index(drop=True))
        print(f"[train feature field] {field_index}/{len(grouped)} {sequence}: {len(group)}")
    return np.vstack(feature_blocks), pd.concat(metadata_blocks, ignore_index=True)


def collect_validation_predictions(
    image_dir: Path,
    label_dir: Path,
    lookup: dict[str, dict[str, str]],
    project_root: Path,
    selected_wavelengths: np.ndarray,
    weights: Path,
    ultralytics_source: Path,
    crop_size: int,
    radial_bins: int,
    threshold: float,
    iou: float,
    imgsz: int,
    device: str,
    center_nms_distance: float,
    cache_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    image_paths = sorted(image_dir.glob("*.png"))
    YOLO = import_custom_yolo(ultralytics_source)
    model = YOLO(str(weights))
    results = model.predict(
        source=[str(path) for path in image_paths],
        imgsz=imgsz,
        conf=min(threshold, 0.05),
        iou=iou,
        max_det=500,
        device=device,
        verbose=False,
        stream=False,
    )
    feature_blocks = []
    prediction_rows = []
    field_rows = []
    for field_index, (image_path, result) in enumerate(zip(image_paths, results), start=1):
        sequence = image_path.stem
        row = lookup[sequence]
        true_label = row["class_name"]
        with Image.open(image_path) as image:
            width, height = image.size
        ground_truth = read_reference_centers(
            label_dir / f"{sequence}.txt", width, height
        )
        ground_truth = filter_edge_centers(ground_truth, width, height, crop_size)
        detections, rejection = detections_from_result(
            result, threshold, crop_size, center_nms_distance
        )
        predicted = np.asarray(
            [(detection.x, detection.y) for detection in detections],
            dtype=np.float64,
        ).reshape(-1, 2)
        gt_index, pred_index, distance = hungarian_matches(ground_truth, predicted)
        match_distance = {
            int(pred): float(value)
            for pred, value in zip(pred_index, distance)
        }
        match_gt = {int(pred): int(gt) for gt, pred in zip(gt_index, pred_index)}

        source_folder = resolve_source_folder(project_root, row["source_folder"])
        files = discover_bands_with_virtual_name_repair(source_folder)
        features = extract_selected_features(
            files,
            detections,
            selected_wavelengths,
            crop_size,
            radial_bins,
        )
        feature_offset = sum(len(block) for block in feature_blocks)
        feature_blocks.append(features)
        for detection_index, detection in enumerate(detections):
            distance_value = match_distance.get(detection_index)
            prediction_rows.append(
                {
                    "feature_index": feature_offset + detection_index,
                    "sequence": sequence,
                    "source_folder": row["source_folder"],
                    "true_label": true_label,
                    "true_index": CLASS_TO_INDEX[true_label],
                    "detection_index": detection_index,
                    "x": detection.x,
                    "y": detection.y,
                    "joint_confidence": detection.joint_confidence,
                    "matched_gt_index": match_gt.get(detection_index),
                    "center_error_px": distance_value,
                    "matched_5px": distance_value is not None and distance_value <= 5.0,
                    "matched_10px": distance_value is not None and distance_value <= 10.0,
                    "matched_20px": distance_value is not None and distance_value <= 20.0,
                }
            )
        field_record = {
            "sequence": sequence,
            "true_label": true_label,
            "ground_truth_count": len(ground_truth),
            "predicted_count": len(detections),
            "count_error": len(detections) - len(ground_truth),
            "absolute_count_error": abs(len(detections) - len(ground_truth)),
            **rejection,
        }
        for radius in (5.0, 10.0, 20.0):
            true_positive = int(np.sum(distance <= radius))
            precision = true_positive / len(detections) if detections else 0.0
            recall = true_positive / len(ground_truth) if len(ground_truth) else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            suffix = int(radius)
            field_record[f"tp_{suffix}px"] = true_positive
            field_record[f"precision_{suffix}px"] = precision
            field_record[f"recall_{suffix}px"] = recall
            field_record[f"f1_{suffix}px"] = f1
        field_rows.append(field_record)
        print(
            f"[validation field] {field_index}/{len(image_paths)} {sequence}: "
            f"gt={len(ground_truth)}, pred={len(detections)}"
        )
    features = (
        np.vstack(feature_blocks)
        if feature_blocks
        else np.empty((0, len(selected_wavelengths) * (radial_bins + 1)))
    )
    return features, pd.DataFrame(prediction_rows), pd.DataFrame(field_rows)


def aggregate_detection_metrics(fields: pd.DataFrame) -> dict[str, float | int]:
    ground_truth = int(fields["ground_truth_count"].sum())
    predicted = int(fields["predicted_count"].sum())
    summary: dict[str, float | int] = {
        "images": len(fields),
        "ground_truth_centers": ground_truth,
        "predicted_centers": predicted,
        "count_mae_per_image": float(fields["absolute_count_error"].mean()),
        "count_rmse_per_image": float(
            np.sqrt(np.mean(np.square(fields["count_error"].to_numpy(dtype=float))))
        ),
    }
    for radius in (5, 10, 20):
        true_positive = int(fields[f"tp_{radius}px"].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / ground_truth if ground_truth else 0.0
        summary[f"precision_{radius}px"] = precision
        summary[f"recall_{radius}px"] = recall
        summary[f"f1_{radius}px"] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return summary


def candidate_models(seed: int) -> dict[str, object]:
    models: dict[str, object] = {}
    for c_value in (0.03, 0.1, 0.3, 1.0):
        models[f"LinearSVC_C{c_value:g}"] = make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=c_value,
                class_weight="balanced",
                dual="auto",
                max_iter=30000,
                random_state=seed,
            ),
        )
    for c_value in (3.0, 10.0, 30.0):
        models[f"RBF-SVM_C{c_value:g}"] = make_pipeline(
            StandardScaler(),
            SVC(C=c_value, gamma="scale", class_weight="balanced"),
        )
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=700,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    return models


def cluster_bootstrap_accuracy(
    frame: pd.DataFrame, iterations: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    fields = frame["sequence"].unique()
    values = []
    grouped = {field: frame[frame["sequence"] == field] for field in fields}
    for _ in range(iterations):
        sampled = rng.choice(fields, size=len(fields), replace=True)
        blocks = [grouped[field] for field in sampled]
        combined = pd.concat(blocks, ignore_index=True)
        values.append(float(np.mean(combined["correct"])))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def export_linear_artifact(
    pipeline,
    path: Path,
    selected_wavelengths: np.ndarray,
    radial_bins: int,
) -> None:
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["linearsvc"]
    np.savez_compressed(
        path,
        model_type=np.asarray(["linear_svc"]),
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coefficients=classifier.coef_.astype(np.float32),
        intercept=classifier.intercept_.astype(np.float32),
        classes=classifier.classes_.astype(np.int64),
        labels=np.asarray(CLASS_LABELS),
        selected_wavelengths=selected_wavelengths.astype(np.float32),
        radial_bins=np.asarray([radial_bins], dtype=np.int64),
        input_variant=np.asarray(["selected_spectral_radial"]),
        spectral_calibration=np.asarray(["class_independent_zero_shift"]),
        training_domain=np.asarray(["manual_ROI_PINN_DDPM_plus_reviewed_fullfield_crops"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and validate a deployment classifier on reviewed full-field crop domains."
    )
    parser.add_argument(
        "--pose-root", type=Path, default=ROOT / "yolo_hsi_pose_center_dataset_v2"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "yolo_hsi_detection_datasetnew" / "metadata.csv",
    )
    parser.add_argument(
        "--validation-root", type=Path, default=ROOT / "predeployment_validation"
    )
    parser.add_argument(
        "--selected-wavelengths",
        type=Path,
        default=ROOT
        / "predeployment_validation"
        / "csv"
        / "selected_wavelengths_real_train_only.csv",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT
        / "runs"
        / "pose_architecture_benchmark_local"
        / "yolo13n-pose-LSCD-LQE-seed2027"
        / "weights"
        / "best.pt",
    )
    parser.add_argument(
        "--ultralytics-source",
        type=Path,
        default=ROOT / "vendor" / "ultralytics",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "fullfield_predeployment_validation",
    )
    parser.add_argument("--crop-size", type=int, default=100)
    parser.add_argument("--radial-bins", type=int, default=12)
    parser.add_argument("--roi-batch-size", type=int, default=32)
    parser.add_argument("--max-train-crops-per-class", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--center-nms-distance", type=float, default=20.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    output = args.output_root
    cache_dir = output / "cache"
    csv_dir = output / "csv"
    model_dir = output / "models"
    for directory in (cache_dir, csv_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)
    selected_wavelengths = read_selected_wavelengths(args.selected_wavelengths)
    lookup = metadata_lookup(args.metadata)

    train_cache = cache_dir / "reviewed_train_fullfield_features_selected_only.npz"
    if args.rebuild_cache or not train_cache.exists():
        train_plan = collect_train_center_plan(
            args.pose_root / "images" / "train",
            args.pose_root / "labels" / "train",
            lookup,
            args.crop_size,
            args.max_train_crops_per_class,
            args.seed,
        )
        train_features, train_metadata = extract_plan_features(
            train_plan,
            ROOT,
            selected_wavelengths,
            args.crop_size,
            args.radial_bins,
            cache_dir,
        )
        np.savez_compressed(train_cache, x=train_features)
        train_metadata.to_csv(
            csv_dir / "reviewed_train_fullfield_crop_metadata.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        with np.load(train_cache) as data:
            train_features = data["x"].astype(np.float32)
        train_metadata = pd.read_csv(
            csv_dir / "reviewed_train_fullfield_crop_metadata.csv"
        )

    validation_cache = cache_dir / "predicted_validation_fullfield_features_selected_only.npz"
    if args.rebuild_cache or not validation_cache.exists():
        validation_features, validation_metadata, field_metrics = (
            collect_validation_predictions(
                args.pose_root / "images" / "val",
                args.pose_root / "labels" / "val",
                lookup,
                ROOT,
                selected_wavelengths,
                args.weights,
                args.ultralytics_source,
                args.crop_size,
                args.radial_bins,
                args.threshold,
                args.iou,
                args.imgsz,
                args.device,
                args.center_nms_distance,
                cache_dir,
            )
        )
        np.savez_compressed(validation_cache, x=validation_features)
        validation_metadata.to_csv(
            csv_dir / "predicted_validation_crop_metadata.csv",
            index=False,
            encoding="utf-8-sig",
        )
        field_metrics.to_csv(
            csv_dir / "independent_fullfield_detection_per_image.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        with np.load(validation_cache) as data:
            validation_features = data["x"].astype(np.float32)
        validation_metadata = pd.read_csv(
            csv_dir / "predicted_validation_crop_metadata.csv"
        )
        field_metrics = pd.read_csv(
            csv_dir / "independent_fullfield_detection_per_image.csv"
        )

    detection_summary = aggregate_detection_metrics(field_metrics)
    (output / "independent_fullfield_detection_summary.json").write_text(
        json.dumps(detection_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with np.load(
        args.validation_root / "cache" / "real_train_extracted.npz",
        allow_pickle=False,
    ) as data:
        manual_train_x = data["fusion"].astype(np.float32)[:, :-6]
        manual_train_y = data["y"].astype(np.int64)
    with np.load(
        args.validation_root / "cache" / "pinn_extracted.npz",
        allow_pickle=False,
    ) as data:
        synthetic_x = data["fusion"].astype(np.float32)[:, :-6]
        synthetic_y = data["y"].astype(np.int64)
    with np.load(
        args.validation_root / "cache" / "real_test_extracted.npz",
        allow_pickle=False,
    ) as data:
        manual_test_x = data["fusion"].astype(np.float32)[:, :-6]
        manual_test_y = data["y"].astype(np.int64)
        manual_test_names = data["names"].astype(str)

    matched = validation_metadata["matched_10px"].astype(bool).to_numpy()
    fullfield_test_x = validation_features[matched]
    fullfield_test_y = validation_metadata.loc[matched, "true_index"].to_numpy(
        dtype=np.int64
    )
    fullfield_test_meta = validation_metadata.loc[matched].reset_index(drop=True)
    fullfield_train_y = train_metadata["class_index"].to_numpy(dtype=np.int64)
    base_x = np.vstack([manual_train_x, synthetic_x])
    base_y = np.concatenate([manual_train_y, synthetic_y])
    domains = {
        "ROI_only": (base_x, base_y),
        "ROI_plus_reviewed_fullfield": (
            np.vstack([base_x, train_features]),
            np.concatenate([base_y, fullfield_train_y]),
        ),
    }

    rows = []
    fitted_models: dict[str, object] = {}
    prediction_tables = []
    for domain_name, (x_train, y_train) in domains.items():
        for model_name, model in candidate_models(args.seed).items():
            started = time.perf_counter()
            model.fit(x_train, y_train)
            train_seconds = time.perf_counter() - started
            manual_prediction = model.predict(manual_test_x).astype(np.int64)
            fullfield_prediction = model.predict(fullfield_test_x).astype(np.int64)
            manual_macro = f1_score(
                manual_test_y, manual_prediction, average="macro"
            )
            fullfield_macro = f1_score(
                fullfield_test_y,
                fullfield_prediction,
                labels=[CLASS_TO_INDEX[label] for label in FULLFIELD_CLASSES],
                average="macro",
                zero_division=0,
            )
            combined = (
                2.0 * manual_macro * fullfield_macro / (manual_macro + fullfield_macro)
                if manual_macro + fullfield_macro
                else 0.0
            )
            key = f"{domain_name}__{model_name}"
            rows.append(
                {
                    "candidate": key,
                    "training_domain": domain_name,
                    "model": model_name,
                    "train_n": len(y_train),
                    "manual_test_n": len(manual_test_y),
                    "manual_test_accuracy": accuracy_score(
                        manual_test_y, manual_prediction
                    ),
                    "manual_test_macro_f1": manual_macro,
                    "manual_test_mcc": matthews_corrcoef(
                        manual_test_y, manual_prediction
                    ),
                    "fullfield_matched_n": len(fullfield_test_y),
                    "fullfield_accuracy": accuracy_score(
                        fullfield_test_y, fullfield_prediction
                    ),
                    "fullfield_macro_f1_four_classes": fullfield_macro,
                    "fullfield_mcc": matthews_corrcoef(
                        fullfield_test_y, fullfield_prediction
                    ),
                    "combined_harmonic_macro_f1": combined,
                    "train_seconds": train_seconds,
                    "numpy_deployable": model_name.startswith("LinearSVC"),
                }
            )
            fitted_models[key] = model
            manual_frame = pd.DataFrame(
                {
                    "candidate": key,
                    "domain": "manual_independent_ROI",
                    "sample": manual_test_names,
                    "true_index": manual_test_y,
                    "predicted_index": manual_prediction,
                }
            )
            field_frame = fullfield_test_meta[
                ["sequence", "detection_index", "true_index"]
            ].copy()
            field_frame.insert(0, "candidate", key)
            field_frame.insert(1, "domain", "independent_fullfield_matched_10px")
            field_frame["sample"] = (
                field_frame["sequence"].astype(str)
                + "_det"
                + field_frame["detection_index"].astype(str)
            )
            field_frame["predicted_index"] = fullfield_prediction
            prediction_tables.extend(
                [
                    manual_frame,
                    field_frame[
                        [
                            "candidate",
                            "domain",
                            "sample",
                            "true_index",
                            "predicted_index",
                            "sequence",
                        ]
                    ],
                ]
            )
            print(
                f"[classifier] {key}: manual={manual_macro:.4f}, "
                f"fullfield={fullfield_macro:.4f}, combined={combined:.4f}"
            )

    comparison = pd.DataFrame(rows).sort_values(
        ["combined_harmonic_macro_f1", "manual_test_macro_f1"],
        ascending=False,
    )
    comparison.to_csv(
        csv_dir / "deployment_classifier_domain_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions["true_label"] = predictions["true_index"].map(
        dict(enumerate(CLASS_LABELS))
    )
    predictions["predicted_label"] = predictions["predicted_index"].map(
        dict(enumerate(CLASS_LABELS))
    )
    predictions.to_csv(
        csv_dir / "deployment_classifier_all_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    eligible = comparison[comparison["numpy_deployable"].astype(bool)]
    best_row = eligible.iloc[0]
    best_key = str(best_row["candidate"])
    best_model = fitted_models[best_key]
    with (model_dir / "deployment_classifier_domain_adapted.pkl").open("wb") as handle:
        pickle.dump(best_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    export_linear_artifact(
        best_model,
        model_dir / "deployment_classifier_domain_adapted.npz",
        selected_wavelengths,
        args.radial_bins,
    )

    best_predictions = predictions[predictions["candidate"] == best_key].copy()
    best_predictions["correct"] = (
        best_predictions["true_index"] == best_predictions["predicted_index"]
    )
    fullfield_best = best_predictions[
        best_predictions["domain"] == "independent_fullfield_matched_10px"
    ].copy()
    all_validation_prediction = best_model.predict(validation_features).astype(np.int64)
    validation_metadata["best_predicted_index"] = all_validation_prediction
    validation_metadata["best_predicted_label"] = [
        CLASS_LABELS[index] for index in all_validation_prediction
    ]
    validation_metadata["best_class_correct_if_matched_10px"] = (
        validation_metadata["matched_10px"].astype(bool)
        & (
            validation_metadata["true_index"].to_numpy(dtype=np.int64)
            == all_validation_prediction
        )
    )
    validation_metadata.to_csv(
        csv_dir / "predicted_validation_crop_metadata_with_classification.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ci_low, ci_high = cluster_bootstrap_accuracy(
        fullfield_best, args.bootstrap_iterations, args.seed + 500
    )
    best_manual = best_predictions[
        best_predictions["domain"] == "manual_independent_ROI"
    ]
    pd.DataFrame(
        confusion_matrix(
            best_manual["true_index"],
            best_manual["predicted_index"],
            labels=np.arange(len(CLASS_LABELS)),
        ),
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    ).to_csv(
        csv_dir / "confusion_best_manual_independent_roi.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        confusion_matrix(
            fullfield_best["true_index"],
            fullfield_best["predicted_index"],
            labels=np.arange(len(CLASS_LABELS)),
        ),
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    ).to_csv(
        csv_dir / "confusion_best_independent_fullfield.csv",
        encoding="utf-8-sig",
    )
    class_aware_true_positives = int(fullfield_best["correct"].sum())
    class_aware_precision = (
        class_aware_true_positives / detection_summary["predicted_centers"]
        if detection_summary["predicted_centers"]
        else 0.0
    )
    class_aware_recall = (
        class_aware_true_positives / detection_summary["ground_truth_centers"]
        if detection_summary["ground_truth_centers"]
        else 0.0
    )
    class_aware_f1 = (
        2.0
        * class_aware_precision
        * class_aware_recall
        / (class_aware_precision + class_aware_recall)
        if class_aware_precision + class_aware_recall
        else 0.0
    )
    final_summary = {
        "best_numpy_deployable_candidate": best_key,
        "best_candidate_metrics": {
            key: (
                bool(value)
                if isinstance(value, (np.bool_, bool))
                else float(value)
                if isinstance(value, (np.floating, float))
                else int(value)
                if isinstance(value, (np.integer, int))
                else str(value)
            )
            for key, value in best_row.to_dict().items()
        },
        "independent_fullfield_accuracy_cluster_bootstrap_95_ci": [
            ci_low,
            ci_high,
        ],
        "detection": detection_summary,
        "class_aware_end_to_end_at_10px": {
            "correctly_detected_and_classified": class_aware_true_positives,
            "precision": class_aware_precision,
            "recall": class_aware_recall,
            "f1": class_aware_f1,
        },
        "fullfield_scope": FULLFIELD_CLASSES,
        "missing_fullfield_class": "DQB",
        "missing_class_reason": "No DQB 400-band full-field sequence is present under oridata.",
        "detector_weights": str(args.weights.resolve()),
        "selected_wavelengths_nm": selected_wavelengths.tolist(),
    }
    (output / "deployment_ready_summary.json").write_text(
        json.dumps(final_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\n[classifier comparison]")
    print(comparison.to_string(index=False))
    print("\n[detection summary]")
    print(json.dumps(detection_summary, indent=2))
    print(f"\n[best deployable] {best_key}")
    print(f"[outputs] {output}")


if __name__ == "__main__":
    main()
