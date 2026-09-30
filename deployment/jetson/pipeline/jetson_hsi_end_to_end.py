from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "shili"
DEFAULT_OUTPUT = ROOT / "jetson_pipeline_results" / "shili"
DEFAULT_ULTRALYTICS_SOURCE = ROOT.parent.parent / "vendor" / "ultralytics"
DEFAULT_DETECTOR = (
    ROOT
    / "runs"
    / "pose_architecture_benchmark_local"
    / "yolo13n-pose-LSCD-LQE-seed2027"
    / "weights"
    / "best.pt"
)
DEFAULT_CLASSIFIER_PICKLE = (
    ROOT
    / "fullfield_predeployment_validation"
    / "models"
    / "deployment_classifier_domain_adapted.pkl"
)
DEFAULT_CLASSIFIER_NUMPY = (
    ROOT
    / "fullfield_predeployment_validation"
    / "models"
    / "deployment_classifier_domain_adapted.npz"
)
DEFAULT_SELECTED_WAVELENGTHS = (
    ROOT
    / "predeployment_validation"
    / "csv"
    / "selected_wavelengths_real_train_only.csv"
)

CLASS_LABELS = ["CMB", "DQB", "DWB", "WQ", "YMXB"]
CLASS_NAMES_ZH = {
    "CMB": "小麦赤霉病分生孢子",
    "DQB": "稻曲病厚垣孢子",
    "DWB": "稻瘟病分生孢子",
    "WQ": "10 µm PVC微球",
    "YMXB": "南方玉米锈病孢子",
}
CLASS_COLORS = {
    "CMB": (226, 78, 78),
    "DQB": (236, 145, 52),
    "DWB": (44, 137, 202),
    "WQ": (41, 162, 98),
    "YMXB": (137, 88, 191),
}

POSITION_RE = re.compile(r"pos_(-?[0-9]+)$")
MODEL_WAVELENGTHS = np.arange(411.0, 780.0, 1.0, dtype=np.float64)
DEFAULT_CONCENTRATION_SLOPE = 7177.958896475087
DEFAULT_CONCENTRATION_INTERCEPT = 16.595417284901487
DEFAULT_CONCENTRATION_CALIBRATION = "final_six_gradient_10um_pvc_20260728"


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    box_confidence: float
    keypoint_confidence: float
    joint_confidence: float


def parse_position(path: Path) -> int:
    match = POSITION_RE.fullmatch(path.stem)
    if not match:
        raise ValueError(f"Cannot parse stage position from {path.name}")
    return int(match.group(1))


def discover_bands(folder: Path) -> list[Path]:
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ]
    invalid = [path.name for path in files if POSITION_RE.fullmatch(path.stem) is None]
    if invalid:
        raise ValueError(f"Image names must follow pos_<integer>.*; invalid examples: {invalid[:5]}")
    files.sort(key=parse_position)
    if not files:
        raise FileNotFoundError(f"No pos_<integer> images found in {folder}")
    positions = np.asarray([parse_position(path) for path in files], dtype=np.int64)
    if len(np.unique(positions)) != len(positions):
        raise ValueError("Duplicate stage positions were found")
    if len(files) > 2:
        steps = np.diff(positions)
        if not np.all(steps == steps[0]):
            raise ValueError(f"Stage positions are not uniformly spaced: {sorted(set(steps.tolist()))[:8]}")
    return files


def wavelength_axis(files: list[Path], wavelength_min: float, wavelength_max: float) -> np.ndarray:
    return np.linspace(wavelength_min, wavelength_max, len(files), dtype=np.float64)


def validate_image_shapes(files: list[Path]) -> tuple[int, int]:
    with Image.open(files[0]) as image:
        width, height = image.size
    for path in files[1:]:
        with Image.open(path) as image:
            if image.size != (width, height):
                raise ValueError(f"Image-size mismatch at {path}: {image.size} != {(width, height)}")
    return width, height


def read_gray_u8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def robust_norm(image: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.uint8)
    low = float(np.percentile(finite, low_pct))
    high = float(np.percentile(finite, high_pct))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.uint8)
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def clahe_channel(image_u8: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_u8)


def choose_pca_files(files: list[Path], pca_bands: int) -> list[Path]:
    if pca_bands <= 1 or pca_bands >= len(files):
        return files
    indices = np.unique(np.linspace(0, len(files) - 1, pca_bands).round().astype(int))
    return [files[int(index)] for index in indices]


def pca_first_component(files: list[Path], sample_pixels: int, seed: int) -> np.ndarray:
    stack = np.stack([read_gray_u8(path) for path in files], axis=0).astype(np.float32)
    bands, height, width = stack.shape
    flat = stack.reshape(bands, -1)
    rng = np.random.default_rng(seed)
    if 0 < sample_pixels < flat.shape[1]:
        sample_indices = rng.choice(flat.shape[1], size=sample_pixels, replace=False)
        sample = flat[:, sample_indices]
    else:
        sample = flat
    means = sample.mean(axis=1, keepdims=True)
    stds = sample.std(axis=1, keepdims=True) + 1e-6
    sample_z = (sample - means) / stds
    covariance = (sample_z @ sample_z.T) / max(1, sample_z.shape[1] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    weights = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    if weights.sum() < 0:
        weights *= -1.0
    full_z = (flat - means) / stds
    return (weights @ full_z).reshape(height, width).astype(np.float32)


def fringe_energy(image: np.ndarray) -> np.ndarray:
    image_u8 = robust_norm(image)
    enhanced = clahe_channel(image_u8).astype(np.float32)
    background = cv2.GaussianBlur(enhanced, ksize=(0, 0), sigmaX=18.0, sigmaY=18.0)
    highpass = enhanced - background
    laplacian = cv2.Laplacian(
        cv2.GaussianBlur(enhanced, ksize=(0, 0), sigmaX=1.2), cv2.CV_32F, ksize=3
    )
    return 0.65 * np.abs(highpass) + 0.35 * np.abs(laplacian)


def build_pseudo_rgb(
    files: list[Path],
    wavelength_min: float,
    wavelength_max: float,
    detection_wavelength: float,
    pca_bands: int,
    pca_sample_pixels: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    wavelengths = wavelength_axis(files, wavelength_min, wavelength_max)
    detection_index = int(np.argmin(np.abs(wavelengths - detection_wavelength)))
    detection_file = files[detection_index]
    band_image = read_gray_u8(detection_file).astype(np.float32)
    pca_files = choose_pca_files(files, pca_bands)
    pc1 = pca_first_component(pca_files, pca_sample_pixels, seed)
    fringe = fringe_energy(band_image)
    red = clahe_channel(robust_norm(band_image))
    green = clahe_channel(robust_norm(pc1, low_pct=0.5, high_pct=99.5))
    blue = clahe_channel(robust_norm(fringe, low_pct=1.0, high_pct=99.7))
    pseudo_rgb = np.dstack([red, green, blue]).astype(np.uint8)
    metadata = {
        "red_channel": "CLAHE-normalized single band",
        "green_channel": "CLAHE-normalized PCA first component",
        "blue_channel": "CLAHE-normalized diffraction-fringe energy",
        "detection_file": detection_file.name,
        "requested_detection_wavelength_nm": detection_wavelength,
        "actual_detection_wavelength_nm": float(wavelengths[detection_index]),
        "pca_bands": len(pca_files),
        "pca_files": [path.name for path in pca_files],
        "pca_seed": seed,
        "width": int(pseudo_rgb.shape[1]),
        "height": int(pseudo_rgb.shape[0]),
    }
    return pseudo_rgb, metadata


def import_custom_yolo(source_root: Path):
    source_root = source_root.resolve()
    if (source_root / "ultralytics" / "__init__.py").exists():
        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
    from ultralytics import YOLO

    return YOLO


def suppress_near_duplicate_centers(detections: list[Detection], min_distance: float) -> list[Detection]:
    kept: list[Detection] = []
    min_distance_sq = min_distance * min_distance
    for item in sorted(detections, key=lambda value: value.joint_confidence, reverse=True):
        if all((item.x - other.x) ** 2 + (item.y - other.y) ** 2 >= min_distance_sq for other in kept):
            kept.append(item)
    return sorted(kept, key=lambda value: (value.y, value.x))


def read_reference_centers(path: Path, width: int, height: int) -> np.ndarray:
    centers: list[tuple[float, float]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) >= 8 and values[7] > 0:
            x, y = values[5], values[6]
        elif len(values) >= 5:
            x, y = values[1], values[2]
        else:
            raise ValueError(f"Invalid YOLO label at {path}:{line_number}")
        if abs(x) <= 1.5 and abs(y) <= 1.5:
            x *= width
            y *= height
        centers.append((x, y))
    return np.asarray(centers, dtype=np.float64).reshape(-1, 2)


def evaluate_reference_centers(
    detections: list[Detection],
    predicted_labels: list[str],
    reference_centers: np.ndarray,
    match_radius_px: float,
    expected_class_label: str | None,
) -> dict[str, object]:
    predicted_centers = np.asarray([(item.x, item.y) for item in detections], dtype=np.float64).reshape(-1, 2)
    candidates: list[tuple[float, int, int]] = []
    for reference_index, reference in enumerate(reference_centers):
        for detection_index, prediction in enumerate(predicted_centers):
            distance = float(np.linalg.norm(reference - prediction))
            if distance <= match_radius_px:
                candidates.append((distance, reference_index, detection_index))
    matched_reference: set[int] = set()
    matched_detections: set[int] = set()
    matches: list[tuple[float, int, int]] = []
    for distance, reference_index, detection_index in sorted(candidates):
        if reference_index in matched_reference or detection_index in matched_detections:
            continue
        matched_reference.add(reference_index)
        matched_detections.add(detection_index)
        matches.append((distance, reference_index, detection_index))

    true_positives = len(matches)
    false_positives = len(detections) - true_positives
    false_negatives = len(reference_centers) - true_positives
    precision = true_positives / len(detections) if detections else 0.0
    recall = true_positives / len(reference_centers) if len(reference_centers) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    distances = np.asarray([item[0] for item in matches], dtype=np.float64)
    result: dict[str, object] = {
        "match_radius_px": match_radius_px,
        "reference_count": int(len(reference_centers)),
        "predicted_count": len(detections),
        "count_error": len(detections) - int(len(reference_centers)),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "center_error_mean_px": float(distances.mean()) if distances.size else None,
        "center_error_median_px": float(np.median(distances)) if distances.size else None,
        "center_error_max_px": float(distances.max()) if distances.size else None,
        "matched_pairs": [
            {
                "reference_index": reference_index + 1,
                "roi_id": detection_index + 1,
                "distance_px": distance,
                "predicted_label": predicted_labels[detection_index],
            }
            for distance, reference_index, detection_index in matches
        ],
        "unmatched_reference_indices": [
            index + 1 for index in range(len(reference_centers)) if index not in matched_reference
        ],
        "unmatched_roi_ids": [index + 1 for index in range(len(detections)) if index not in matched_detections],
    }
    if expected_class_label is not None:
        correct = sum(predicted_labels[detection_index] == expected_class_label for _, _, detection_index in matches)
        result["expected_class_label"] = expected_class_label
        result["matched_classification_correct"] = correct
        result["matched_classification_accuracy"] = correct / true_positives if true_positives else None
    return result


def run_pose_detector(
    pseudo_rgb_path: Path,
    weights: Path,
    ultralytics_source: Path,
    imgsz: int,
    threshold: float,
    iou: float,
    max_detections: int,
    device: str,
    crop_size: int,
    center_nms_distance: float,
) -> tuple[list[Detection], dict[str, object]]:
    YOLO = import_custom_yolo(ultralytics_source)
    model = YOLO(str(weights), task="pose")
    results = model.predict(
        source=str(pseudo_rgb_path),
        imgsz=imgsz,
        conf=min(threshold, 0.05),
        iou=iou,
        max_det=max_detections,
        device=device,
        verbose=False,
    )
    result = results[0]
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or len(boxes) == 0:
        return [], {"raw_predictions": 0, "confidence_rejected": 0, "edge_rejected": 0, "duplicate_rejected": 0}
    centers = keypoints.xy[:, 0, :].detach().cpu().numpy()
    box_scores = boxes.conf.detach().cpu().numpy()
    if keypoints.conf is None:
        keypoint_scores = np.ones_like(box_scores)
    else:
        keypoint_scores = keypoints.conf[:, 0].detach().cpu().numpy()
    height, width = result.orig_shape
    edge_margin = crop_size / 2.0
    candidates: list[Detection] = []
    confidence_rejected = 0
    edge_rejected = 0
    for center, box_score, keypoint_score in zip(centers, box_scores, keypoint_scores):
        x, y = float(center[0]), float(center[1])
        if not np.isfinite([x, y, box_score, keypoint_score]).all():
            confidence_rejected += 1
            continue
        if box_score < threshold or keypoint_score < threshold:
            confidence_rejected += 1
            continue
        if x < edge_margin or y < edge_margin or x > width - edge_margin or y > height - edge_margin:
            edge_rejected += 1
            continue
        candidates.append(
            Detection(
                x=int(round(x)),
                y=int(round(y)),
                box_confidence=float(box_score),
                keypoint_confidence=float(keypoint_score),
                joint_confidence=float(min(box_score, keypoint_score)),
            )
        )
    detections = suppress_near_duplicate_centers(candidates, center_nms_distance)
    metadata = {
        "backend": weights.suffix.lower().lstrip("."),
        "model_task": str(model.task),
        "model_names": {str(key): value for key, value in model.names.items()},
        "raw_predictions": int(len(boxes)),
        "confidence_rejected": confidence_rejected,
        "edge_rejected": edge_rejected,
        "duplicate_rejected": len(candidates) - len(detections),
        "accepted_detections": len(detections),
        "threshold": threshold,
        "iou": iou,
        "imgsz": imgsz,
        "device": device,
    }
    return detections, metadata


def read_selected_wavelengths(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "wavelength_nm" not in rows[0]:
        raise ValueError(f"Expected a wavelength_nm column in {path}")
    return np.asarray([float(row["wavelength_nm"]) for row in rows], dtype=np.float64)


def export_linear_svm_numpy(
    pickle_path: Path,
    npz_path: Path,
    selected_wavelengths: np.ndarray,
    radial_bins: int,
) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pickle_path.open("rb") as handle:
            pipeline = pickle.load(handle)
    if not hasattr(pipeline, "named_steps"):
        raise TypeError("The classifier pickle is not an sklearn Pipeline")
    scaler = pipeline.named_steps.get("standardscaler")
    classifier = pipeline.named_steps.get("linearsvc")
    if scaler is None or classifier is None:
        raise TypeError(f"Expected standardscaler + linearsvc, found {list(pipeline.named_steps)}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        mean=np.asarray(scaler.mean_, dtype=np.float32),
        scale=np.asarray(scaler.scale_, dtype=np.float32),
        coefficients=np.asarray(classifier.coef_, dtype=np.float32),
        intercept=np.asarray(classifier.intercept_, dtype=np.float32),
        classes=np.asarray(classifier.classes_, dtype=np.int64),
        labels=np.asarray(CLASS_LABELS),
        selected_wavelengths_nm=np.asarray(selected_wavelengths, dtype=np.float32),
        model_wavelengths_nm=np.asarray(MODEL_WAVELENGTHS, dtype=np.float32),
        radial_bins=np.asarray([radial_bins], dtype=np.int64),
        input_variant=np.asarray(["spectral_radial_stats"]),
    )
    rng = np.random.default_rng(20260721)
    test_features = rng.normal(size=(32, len(scaler.mean_))).astype(np.float32)
    sklearn_predictions = pipeline.predict(test_features).astype(np.int64)
    transformed = (test_features - scaler.mean_.astype(np.float32)) / scaler.scale_.astype(np.float32)
    scores = transformed @ classifier.coef_.astype(np.float32).T + classifier.intercept_.astype(np.float32)
    numpy_predictions = classifier.classes_[np.argmax(scores, axis=1)].astype(np.int64)
    exact_match = bool(np.array_equal(sklearn_predictions, numpy_predictions))
    if not exact_match:
        raise RuntimeError("Pure NumPy classifier export did not reproduce sklearn predictions")
    return {
        "source_pickle": str(pickle_path.resolve()),
        "numpy_artifact": str(npz_path.resolve()),
        "feature_dimension": int(len(scaler.mean_)),
        "classes": classifier.classes_.astype(int).tolist(),
        "labels": CLASS_LABELS,
        "validation_random_samples": len(test_features),
        "predictions_exactly_matched": exact_match,
    }


def ensure_classifier_numpy(
    npz_path: Path,
    pickle_path: Path,
    selected_wavelengths: np.ndarray,
    radial_bins: int,
    output_dir: Path,
) -> dict[str, object]:
    if npz_path.exists():
        return {"numpy_artifact": str(npz_path.resolve()), "created_this_run": False}
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Pure NumPy classifier is missing ({npz_path}) and source pickle was not found ({pickle_path})"
        )
    metadata = export_linear_svm_numpy(pickle_path, npz_path, selected_wavelengths, radial_bins)
    metadata["created_this_run"] = True
    (output_dir / "classifier_export_validation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def radial_geometry(crop_size: int, radial_bins: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:crop_size, :crop_size]
    center = (crop_size - 1) / 2.0
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    bin_ids = np.floor(radius / (radius.max() + 1e-6) * radial_bins).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, radial_bins - 1)
    bin_counts = np.bincount(bin_ids.reshape(-1), minlength=radial_bins).astype(np.float32)
    bin_counts[bin_counts == 0] = 1.0
    return bin_ids, bin_counts


def radial_profile(image: np.ndarray, bin_ids: np.ndarray, bin_counts: np.ndarray) -> np.ndarray:
    profile = np.bincount(
        bin_ids.reshape(-1), weights=image.reshape(-1).astype(np.float32), minlength=len(bin_counts)
    )[: len(bin_counts)]
    profile = profile / bin_counts
    maximum = float(np.max(np.abs(profile)))
    if maximum > 0:
        profile = profile / maximum
    return profile.astype(np.float32)


def nonzero_mean(crop: np.ndarray) -> float:
    values = crop[crop > 0]
    return float(values.mean()) if values.size else float(crop.mean())


def write_envi_header(path: Path, raw_name: str, wavelengths: np.ndarray, crop_size: int) -> None:
    wavelength_text = ", ".join(f"{value:.3f}" for value in wavelengths)
    header = f"""ENVI
description = {{Centered hyperspectral diffraction ROI; raw file = {raw_name}}}
samples = {crop_size}
lines = {crop_size}
bands = {len(wavelengths)}
header offset = 0
file type = ENVI Standard
data type = 4
interleave = bsq
sensor type = OV5640
byte order = 0
wavelength units = Nanometers
wavelength = {{{wavelength_text}}}
"""
    path.write_text(header, encoding="ascii")


def extract_features_and_cubes(
    files: list[Path],
    detections: list[Detection],
    selected_wavelengths: np.ndarray,
    wavelength_min: float,
    wavelength_max: float,
    crop_size: int,
    radial_bins: int,
    spectral_shift_bands: float,
    cubes_dir: Path,
    save_cubes: bool,
    circular_mask: bool,
    roi_batch_size: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    feature_dimension = len(selected_wavelengths) * (radial_bins + 1) + 6
    if not detections:
        return np.empty((0, feature_dimension), dtype=np.float32), []
    raw_wavelengths = wavelength_axis(files, wavelength_min, wavelength_max)
    # Match create_corrected_envi.py semantics exactly: first linearly resample
    # the 400 acquired frames to a 1 nm axis, then apply the wavelength shift.
    # A shift of -5 maps corrected 411 nm to acquired 416 nm.
    source_model_wavelengths = MODEL_WAVELENGTHS - spectral_shift_bands
    if source_model_wavelengths.min() < raw_wavelengths.min() or source_model_wavelengths.max() > raw_wavelengths.max():
        raise ValueError(
            "spectral-shift-bands requests wavelengths outside the acquired range: "
            f"{source_model_wavelengths.min():.3f}-{source_model_wavelengths.max():.3f} nm"
        )
    source_positions = np.interp(
        source_model_wavelengths,
        raw_wavelengths,
        np.arange(len(raw_wavelengths), dtype=np.float64),
    )
    lower_indices = np.floor(source_positions).astype(int)
    upper_indices = np.ceil(source_positions).astype(int)
    lower_indices = np.clip(lower_indices, 0, len(files) - 1)
    upper_indices = np.clip(upper_indices, 0, len(files) - 1)
    interpolation_weights = (source_positions - lower_indices).astype(np.float32)
    selected_model_indices = np.asarray(
        [int(np.argmin(np.abs(MODEL_WAVELENGTHS - wavelength))) for wavelength in selected_wavelengths],
        dtype=int,
    )
    selected_orders_by_model: dict[int, list[int]] = {}
    for selected_order, model_index in enumerate(selected_model_indices):
        selected_orders_by_model.setdefault(int(model_index), []).append(selected_order)

    n = len(detections)
    spectra = np.zeros((n, len(MODEL_WAVELENGTHS)), dtype=np.float32)
    selected_spectra = np.zeros((n, len(selected_wavelengths)), dtype=np.float32)
    radial = np.zeros((n, len(selected_wavelengths), radial_bins), dtype=np.float32)
    cube_sum = np.zeros(n, dtype=np.float64)
    cube_sumsq = np.zeros(n, dtype=np.float64)
    bin_ids, bin_counts = radial_geometry(crop_size, radial_bins)
    half = crop_size // 2
    crop_mask = np.ones((crop_size, crop_size), dtype=np.uint8)
    if circular_mask:
        crop_mask.fill(0)
        cv2.circle(crop_mask, (half, half), half, 1, -1)

    cube_maps: list[np.memmap] = []
    cube_records: list[dict[str, object]] = []
    if save_cubes:
        cubes_dir.mkdir(parents=True, exist_ok=True)
        for roi_index in range(n):
            stem = f"roi_{roi_index + 1:04d}"
            raw_path = cubes_dir / f"{stem}.raw"
            header_path = cubes_dir / f"{stem}.hdr"
            cube_map = np.memmap(
                raw_path,
                dtype=np.float32,
                mode="w+",
                shape=(len(MODEL_WAVELENGTHS), crop_size, crop_size),
                order="C",
            )
            cube_maps.append(cube_map)
            write_envi_header(header_path, raw_path.name, MODEL_WAVELENGTHS, crop_size)
            cube_records.append({"roi_id": roi_index + 1, "hdr_path": str(header_path), "raw_path": str(raw_path)})

    if roi_batch_size <= 0:
        raise ValueError("roi-batch-size must be positive")
    for batch_start in range(0, n, roi_batch_size):
        batch_end = min(batch_start + roi_batch_size, n)
        batch_detections = detections[batch_start:batch_end]
        raw_crops = np.empty(
            (len(batch_detections), len(files), crop_size, crop_size),
            dtype=np.uint8,
        )
        for file_index, file_path in enumerate(files):
            image = read_gray_u8(file_path)
            for local_index, detection in enumerate(batch_detections):
                crop = image[
                    detection.y - half : detection.y + half,
                    detection.x - half : detection.x + half,
                ]
                if crop.shape != (crop_size, crop_size):
                    raise RuntimeError(
                        f"Unexpected crop shape {crop.shape} at ROI {batch_start + local_index + 1}"
                    )
                raw_crops[local_index, file_index] = crop * crop_mask

        for model_order, (lower, upper, weight) in enumerate(
            zip(lower_indices, upper_indices, interpolation_weights)
        ):
            lower_crop = raw_crops[:, lower].astype(np.float32)
            if lower == upper or weight == 0.0:
                corrected_crop = lower_crop
            else:
                upper_crop = raw_crops[:, upper].astype(np.float32)
                corrected_crop = lower_crop * (1.0 - weight) + upper_crop * weight

            selected_orders = selected_orders_by_model.get(model_order, [])
            for local_index in range(len(batch_detections)):
                detection_index = batch_start + local_index
                crop = corrected_crop[local_index]
                spectrum_value = nonzero_mean(crop)
                spectra[detection_index, model_order] = spectrum_value
                if save_cubes:
                    cube_maps[detection_index][model_order] = crop
                crop_float = crop.astype(np.float64)
                cube_sum[detection_index] += float(crop_float.sum())
                cube_sumsq[detection_index] += float(np.square(crop_float).sum())
                if selected_orders:
                    profile = radial_profile(crop, bin_ids, bin_counts)
                    for selected_order in selected_orders:
                        selected_spectra[detection_index, selected_order] = spectrum_value
                        radial[detection_index, selected_order] = profile
        print(f"[roi] processed detections {batch_start + 1}-{batch_end}/{n}")

    for cube_map in cube_maps:
        cube_map.flush()
    cube_maps.clear()

    selected_normalized = (selected_spectra - selected_spectra.mean(axis=1, keepdims=True)) / (
        selected_spectra.std(axis=1, keepdims=True) + 1e-6
    )
    cube_pixel_count = float(len(MODEL_WAVELENGTHS) * crop_size * crop_size)
    cube_mean = cube_sum / cube_pixel_count
    cube_variance = np.maximum(cube_sumsq / cube_pixel_count - np.square(cube_mean), 0.0)
    cube_std = np.sqrt(cube_variance)
    global_features = np.column_stack(
        [
            spectra.mean(axis=1),
            spectra.std(axis=1),
            np.percentile(spectra, 10, axis=1),
            np.percentile(spectra, 90, axis=1),
            cube_mean,
            cube_std,
        ]
    ).astype(np.float32)
    features = np.hstack(
        [selected_normalized.astype(np.float32), radial.reshape(n, -1), global_features]
    ).astype(np.float32)
    return features, cube_records


def classify_numpy(features: np.ndarray, artifact_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    artifact = np.load(artifact_path, allow_pickle=False)
    mean = artifact["mean"].astype(np.float32)
    scale = artifact["scale"].astype(np.float32)
    coefficients = artifact["coefficients"].astype(np.float32)
    intercept = artifact["intercept"].astype(np.float32)
    classes = artifact["classes"].astype(np.int64)
    labels = artifact["labels"].astype(str).tolist()
    input_variant = (
        artifact["input_variant"].astype(str).tolist()[0]
        if "input_variant" in artifact.files
        else "spectral_radial_stats"
    )
    scale = np.where(scale == 0, 1.0, scale)
    if (
        input_variant == "selected_spectral_radial"
        and features.shape[1] == len(mean) + 6
    ):
        features = features[:, : len(mean)]
    if features.shape[1] != len(mean):
        raise ValueError(f"Classifier expects {len(mean)} features, received {features.shape[1]}")
    transformed = (features - mean) / scale
    scores = transformed @ coefficients.T + intercept
    order = np.argsort(scores, axis=1)
    winning_columns = order[:, -1]
    predictions = classes[winning_columns]
    margins = scores[np.arange(len(scores)), order[:, -1]] - scores[np.arange(len(scores)), order[:, -2]]
    metadata = {
        "artifact": str(artifact_path.resolve()),
        "input_feature_dimension": int(features.shape[1]),
        "input_variant": input_variant,
        "class_labels": labels,
        "spectral_calibration": (
            artifact["spectral_calibration"].astype(str).tolist()[0]
            if "spectral_calibration" in artifact.files
            else "legacy_class_dependent_alignment"
        ),
        "confidence_definition": "LinearSVC top-1 minus top-2 decision score; not a probability",
    }
    return predictions.astype(int), margins.astype(float), scores.astype(float), metadata


def load_annotation_font(size: int, bold: bool = False):
    font_names = ["msyhbd.ttc", "simhei.ttf"] if bold else ["msyh.ttc", "simsun.ttc"]
    candidates = [Path("C:/Windows/Fonts") / name for name in font_names]
    candidates.extend([Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")])
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def show_stage_window(title: str, image_rgb: np.ndarray, enabled: bool, delay_ms: int) -> None:
    if not enabled:
        return
    try:
        height, width = image_rgb.shape[:2]
        scale = min(1.0, 1400.0 / width, 900.0 / height)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title, max(640, int(width * scale)), max(480, int(height * scale)))
        cv2.imshow(title, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if delay_ms <= 0:
            print(f"[window] {title}: press any key to continue")
            cv2.waitKey(0)
        else:
            cv2.waitKey(delay_ms)
        cv2.destroyWindow(title)
    except cv2.error as error:
        print(f"[window warning] Unable to display '{title}': {error}")


def visualization_background(gray_image: np.ndarray) -> np.ndarray:
    normalized = robust_norm(gray_image, low_pct=0.5, high_pct=99.7)
    enhanced = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(12, 12)).apply(normalized)
    return np.repeat(enhanced[:, :, None], 3, axis=2)


def draw_detection_preview(
    background_rgb: np.ndarray,
    detections: list[Detection],
    crop_size: int,
    output_path: Path,
) -> np.ndarray:
    image = Image.fromarray(background_rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = load_annotation_font(22, bold=True)
    half = crop_size // 2
    color = (31, 205, 108)
    for index, detection in enumerate(detections, start=1):
        draw.rectangle(
            [detection.x - half, detection.y - half, detection.x + half, detection.y + half],
            outline=color,
            width=5,
        )
        draw.ellipse(
            [detection.x - 5, detection.y - 5, detection.x + 5, detection.y + 5],
            fill=(255, 220, 55),
            outline=(20, 20, 20),
            width=2,
        )
    panel_text = f"Detected centers: {len(detections)}"
    panel_box = draw.textbbox((0, 0), panel_text, font=font)
    panel_width = panel_box[2] - panel_box[0] + 32
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([16, 16, 16 + panel_width, 58], fill=(0, 0, 0, 175))
    overlay_draw.text((32, 24), panel_text, fill=(255, 255, 255, 255), font=font)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    image.save(output_path, quality=95)
    return np.asarray(image)


def draw_overlay(
    background_rgb: np.ndarray,
    detections: list[Detection],
    labels: list[str],
    margins: np.ndarray,
    crop_size: int,
    concentration: dict[str, object],
    output_path: Path,
) -> np.ndarray:
    image = Image.fromarray(background_rgb, mode="RGB")
    label_font = load_annotation_font(20, bold=True)
    concentration_font = load_annotation_font(27, bold=True)
    detail_font = load_annotation_font(20, bold=False)
    half = crop_size // 2

    concentration_value = float(concentration["estimated_total_concentration_particles_ml"])
    concentration_text = f"Estimated concentration: {concentration_value:.2e} particles/mL"
    estimate_type = (
        "PVC-calibrated estimate"
        if concentration["mode"] == "empirical_linear_calibration"
        else "Geometric estimate"
    )
    detail_text = f"Detected centers: {len(detections)}    {estimate_type}"
    measure = ImageDraw.Draw(image)
    first_box = measure.textbbox((0, 0), concentration_text, font=concentration_font)
    second_box = measure.textbbox((0, 0), detail_text, font=detail_font)
    panel_width = max(first_box[2] - first_box[0], second_box[2] - second_box[0]) + 42
    panel_height = 92
    panel_rect = (16, 16, 16 + panel_width, 16 + panel_height)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(panel_rect, fill=(0, 0, 0, 165))
    overlay_draw.text((36, 26), concentration_text, fill=(255, 255, 255, 255), font=concentration_font)
    overlay_draw.text((36, 65), detail_text, fill=(225, 229, 234, 255), font=detail_font)
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    object_boxes = [
        (item.x - half, item.y - half, item.x + half, item.y + half) for item in detections
    ]
    used_label_boxes: list[tuple[int, int, int, int]] = [panel_rect]

    def intersects(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
        return not (
            first[2] <= second[0]
            or first[0] >= second[2]
            or first[3] <= second[1]
            or first[1] >= second[3]
        )

    for index, (detection, label, margin) in enumerate(zip(detections, labels, margins), start=1):
        color = CLASS_COLORS[label]
        left = detection.x - half
        top = detection.y - half
        right = detection.x + half
        bottom = detection.y + half
        draw.rectangle(
            [left, top, right, bottom],
            outline=color,
            width=5,
        )
        draw.ellipse(
            [detection.x - 5, detection.y - 5, detection.x + 5, detection.y + 5],
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )
        label_text = f"{label} {index}"
        text_box = draw.textbbox((0, 0), label_text, font=label_font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_width = text_width + 14
        label_height = text_height + 8
        candidates = [
            (left, top - label_height),
            (right - label_width, top - label_height),
            (left, bottom),
            (right - label_width, bottom),
            (left - label_width, top),
            (right, top),
            (left, top + 5),
        ]
        label_rect = None
        for candidate_left, candidate_top in candidates:
            candidate_left = max(0, min(candidate_left, image.width - label_width))
            candidate_top = max(0, min(candidate_top, image.height - label_height))
            candidate = (
                int(candidate_left),
                int(candidate_top),
                int(candidate_left + label_width),
                int(candidate_top + label_height),
            )
            other_boxes = object_boxes[: index - 1] + object_boxes[index:]
            if any(intersects(candidate, obstacle) for obstacle in used_label_boxes + other_boxes):
                continue
            label_rect = candidate
            break
        if label_rect is None:
            fallback_left = max(0, min(left, image.width - label_width))
            fallback_top = max(0, min(top - label_height, image.height - label_height))
            label_rect = (
                int(fallback_left),
                int(fallback_top),
                int(fallback_left + label_width),
                int(fallback_top + label_height),
            )
        used_label_boxes.append(label_rect)
        draw.rectangle(
            label_rect,
            fill=color,
        )
        draw.text(
            (label_rect[0] + 7, label_rect[1] + 2),
            label_text,
            fill=(255, 255, 255),
            font=label_font,
        )
    image.save(output_path, quality=95)
    return np.asarray(image)


def draw_contact_sheet(
    pseudo_rgb: np.ndarray,
    detections: list[Detection],
    labels: list[str],
    margins: np.ndarray,
    crop_size: int,
    output_path: Path,
    max_items: int,
) -> None:
    item_count = min(len(detections), max_items)
    if item_count == 0:
        return
    columns = 8
    rows = int(math.ceil(item_count / columns))
    tile_width = crop_size + 18
    tile_height = crop_size + 32
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(sheet)
    half = crop_size // 2
    for index in range(item_count):
        detection = detections[index]
        crop = pseudo_rgb[
            detection.y - half : detection.y + half,
            detection.x - half : detection.x + half,
        ]
        x0 = (index % columns) * tile_width + 8
        y0 = (index // columns) * tile_height + 6
        sheet.paste(Image.fromarray(crop, mode="RGB"), (x0, y0))
        label = labels[index]
        color = CLASS_COLORS[label]
        draw.rectangle([x0, y0, x0 + crop_size, y0 + crop_size], outline=color, width=2)
        draw.text((x0, y0 + crop_size + 3), f"{index + 1} {label} m={margins[index]:.2f}", fill=color)
    sheet.save(output_path, quality=95)


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def concentration_summary(
    labels: list[str],
    valid_area_mm2: float,
    drop_volume_ul: float,
    wet_area_mm2: float,
    concentration_slope: float | None,
    concentration_intercept: float,
    calibration_source: str,
) -> dict[str, object]:
    effective_field_volume_ul = valid_area_mm2 * drop_volume_ul / wet_area_mm2
    if concentration_slope is None:
        slope = 1000.0 / effective_field_volume_ul
        intercept = 0.0
        mode = "geometric_estimate"
        status = "PRELIMINARY_NOT_EXPERIMENTALLY_CALIBRATED"
    else:
        slope = concentration_slope
        intercept = concentration_intercept
        mode = "empirical_linear_calibration"
        status = "FINAL_SIX_GRADIENT_EMPIRICAL_CALIBRATION"
    if slope <= 0:
        raise ValueError("Calibration slope must be positive")

    def estimate(count: int) -> float:
        if count <= 0:
            return 0.0
        return float(max(0.0, slope * count + intercept))

    counts = {label: labels.count(label) for label in CLASS_LABELS}
    per_class = {
        label: {
            "count": count,
            "estimated_concentration_particles_ml": estimate(count),
            "interpretation": (
                "calibrated with the final six-gradient 10 um PVC experiment"
                if label == "WQ"
                else "total-particle calibration applied; class-specific deposition was not independently calibrated"
            ),
        }
        for label, count in counts.items()
    }
    return {
        "status": status,
        "mode": mode,
        "calibration_source": calibration_source,
        "total_valid_center_count": len(labels),
        "estimated_total_concentration_particles_ml": estimate(len(labels)),
        "count_to_concentration_equation": (
            f"C = 0 for N=0; otherwise C = max(0, {slope:.8g} * N + {intercept:.8g})"
        ),
        "slope_particles_ml_per_count": float(slope),
        "intercept_particles_ml": float(intercept),
        "slope_count_per_particles_ml": float(1.0 / slope),
        "intercept_count": float(-intercept / slope),
        "valid_area_mm2": valid_area_mm2,
        "drop_volume_ul": drop_volume_ul,
        "wet_area_mm2": wet_area_mm2,
        "effective_field_volume_ul": effective_field_volume_ul,
        "per_class": per_class,
    }


def write_band_mapping(
    path: Path, files: list[Path], wavelength_min: float, wavelength_max: float
) -> None:
    wavelengths = wavelength_axis(files, wavelength_min, wavelength_max)
    rows = [
        {
            "band_index": index,
            "position": parse_position(file_path),
            "wavelength_nm": f"{wavelengths[index]:.6f}",
            "filename": file_path.name,
        }
        for index, file_path in enumerate(files)
    ]
    write_rows_csv(path, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end HSI diffraction inference: pseudo-RGB, YOLO13n-Pose-LSCD-LQE center detection, "
            "100x100x369 ROI extraction, five-class classification, counting and concentration estimate."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ultralytics-source", type=Path, default=DEFAULT_ULTRALYTICS_SOURCE)
    parser.add_argument("--detector-weights", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--classifier-numpy", type=Path, default=DEFAULT_CLASSIFIER_NUMPY)
    parser.add_argument("--classifier-pickle", type=Path, default=DEFAULT_CLASSIFIER_PICKLE)
    parser.add_argument("--selected-wavelengths", type=Path, default=DEFAULT_SELECTED_WAVELENGTHS)
    parser.add_argument("--wavelength-min", type=float, default=400.0)
    parser.add_argument("--wavelength-max", type=float, default=800.0)
    parser.add_argument("--expected-bands", type=int, default=400)
    parser.add_argument("--detection-wavelength", type=float, default=480.0)
    parser.add_argument("--pca-bands", type=int, default=24)
    parser.add_argument("--pca-sample-pixels", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        default=[768, 1024],
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--threshold", type=float, default=0.325)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=500)
    parser.add_argument("--center-nms-distance", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--crop-size", type=int, default=100)
    parser.add_argument("--radial-bins", type=int, default=12)
    parser.add_argument(
        "--no-circular-mask",
        action="store_true",
        help="Disable the radius-50 circular ROI mask used by the classifier training data.",
    )
    parser.add_argument(
        "--roi-batch-size",
        type=int,
        default=32,
        help="Number of detected ROIs buffered across all 400 source bands at once.",
    )
    parser.add_argument(
        "--spectral-shift-bands",
        type=float,
        default=0.0,
        help=(
            "Wavelength shift applied before classification, using create_corrected_envi.py semantics. "
            "This is for reproducing legacy corrected data; deployment should use a class-independent calibration."
        ),
    )
    parser.add_argument("--no-save-cubes", action="store_true")
    parser.add_argument("--contact-sheet-items", type=int, default=80)
    parser.add_argument(
        "--show-windows",
        action="store_true",
        help="Show pseudo-RGB, detection, and final-result windows during an interactive local run.",
    )
    parser.add_argument(
        "--popup-delay-ms",
        type=int,
        default=0,
        help="Popup duration in milliseconds; 0 waits for a key press at every stage.",
    )
    parser.add_argument(
        "--reference-pose-label",
        type=Path,
        default=None,
        help="Optional YOLO-Pose label file used only to evaluate a local debug run.",
    )
    parser.add_argument("--match-radius-px", type=float, default=10.0)
    parser.add_argument(
        "--reference-class-label",
        choices=CLASS_LABELS,
        default=None,
        help="Optional known class used to audit classification only on matched ground-truth objects.",
    )
    parser.add_argument("--valid-area-mm2", type=float, default=5.464532592)
    parser.add_argument("--drop-volume-ul", type=float, default=10.0)
    parser.add_argument("--wet-area-mm2", type=float, default=400.0)
    parser.add_argument(
        "--concentration-slope",
        type=float,
        default=DEFAULT_CONCENTRATION_SLOPE,
        help="Slope a in C=a*N+c (particles/mL per detected center).",
    )
    parser.add_argument(
        "--concentration-intercept",
        type=float,
        default=DEFAULT_CONCENTRATION_INTERCEPT,
        help="Intercept c in C=a*N+c (particles/mL).",
    )
    parser.add_argument(
        "--geometric-concentration",
        action="store_true",
        help="Use the field-volume geometric estimate instead of the final six-gradient empirical calibration.",
    )
    parser.add_argument(
        "--calibration-slope",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--calibration-intercept",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.crop_size % 2:
        raise ValueError("crop-size must be even")

    checkpoint = time.perf_counter()
    files = discover_bands(args.input_dir)
    if args.expected_bands > 0 and len(files) != args.expected_bands:
        raise ValueError(f"Expected {args.expected_bands} bands, found {len(files)}")
    width, height = validate_image_shapes(files)
    write_band_mapping(args.output_dir / "band_mapping.csv", files, args.wavelength_min, args.wavelength_max)
    timings["input_validation_seconds"] = time.perf_counter() - checkpoint
    print(f"[input] {len(files)} bands, {width}x{height}, {files[0].name} -> {files[-1].name}")

    checkpoint = time.perf_counter()
    pseudo_rgb, pseudo_metadata = build_pseudo_rgb(
        files,
        args.wavelength_min,
        args.wavelength_max,
        args.detection_wavelength,
        args.pca_bands,
        args.pca_sample_pixels,
        args.seed,
    )
    pseudo_path = args.output_dir / "pseudo_rgb.png"
    Image.fromarray(pseudo_rgb, mode="RGB").save(pseudo_path)
    (args.output_dir / "pseudo_rgb_metadata.json").write_text(
        json.dumps(pseudo_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    timings["pseudo_rgb_seconds"] = time.perf_counter() - checkpoint
    print(f"[pseudo-rgb] {pseudo_path} ({pseudo_metadata['actual_detection_wavelength_nm']:.3f} nm)")
    show_stage_window("Step 1 - Pseudo RGB", pseudo_rgb, args.show_windows, args.popup_delay_ms)
    detection_gray = read_gray_u8(args.input_dir / str(pseudo_metadata["detection_file"]))
    display_background = visualization_background(detection_gray)

    checkpoint = time.perf_counter()
    detections, detector_metadata = run_pose_detector(
        pseudo_path,
        args.detector_weights.resolve(),
        args.ultralytics_source.resolve(),
        args.imgsz,
        args.threshold,
        args.iou,
        args.max_detections,
        args.device,
        args.crop_size,
        args.center_nms_distance,
    )
    timings["pose_detection_seconds"] = time.perf_counter() - checkpoint
    print(
        f"[detect] raw={detector_metadata['raw_predictions']} accepted={len(detections)} "
        f"edge_rejected={detector_metadata['edge_rejected']}"
    )
    detection_preview = draw_detection_preview(
        display_background,
        detections,
        args.crop_size,
        args.output_dir / "detection_preview.png",
    )
    show_stage_window(
        "Step 2 - Diffraction Center Detection",
        detection_preview,
        args.show_windows,
        args.popup_delay_ms,
    )

    selected_wavelengths = read_selected_wavelengths(args.selected_wavelengths.resolve())
    classifier_export_metadata = ensure_classifier_numpy(
        args.classifier_numpy.resolve(),
        args.classifier_pickle.resolve(),
        selected_wavelengths,
        args.radial_bins,
        args.output_dir,
    )

    checkpoint = time.perf_counter()
    features, cube_records = extract_features_and_cubes(
        files,
        detections,
        selected_wavelengths,
        args.wavelength_min,
        args.wavelength_max,
        args.crop_size,
        args.radial_bins,
        args.spectral_shift_bands,
        args.output_dir / "cubes",
        save_cubes=not args.no_save_cubes,
        circular_mask=not args.no_circular_mask,
        roi_batch_size=args.roi_batch_size,
    )
    timings["roi_extraction_and_feature_seconds"] = time.perf_counter() - checkpoint

    checkpoint = time.perf_counter()
    if len(detections):
        predictions, margins, scores, classifier_metadata = classify_numpy(
            features, args.classifier_numpy.resolve()
        )
        predicted_labels = [CLASS_LABELS[int(index)] for index in predictions]
    else:
        predictions = np.empty(0, dtype=int)
        margins = np.empty(0, dtype=float)
        scores = np.empty((0, len(CLASS_LABELS)), dtype=float)
        predicted_labels = []
        classifier_metadata = {
            "artifact": str(args.classifier_numpy.resolve()),
            "input_feature_dimension": 526,
            "class_labels": CLASS_LABELS,
            "confidence_definition": "No detections",
        }
    timings["classification_seconds"] = time.perf_counter() - checkpoint

    rows: list[dict[str, object]] = []
    for index, detection in enumerate(detections):
        row: dict[str, object] = {
            "roi_id": index + 1,
            "x_px": detection.x,
            "y_px": detection.y,
            "box_confidence": f"{detection.box_confidence:.8f}",
            "keypoint_confidence": f"{detection.keypoint_confidence:.8f}",
            "joint_confidence": f"{detection.joint_confidence:.8f}",
            "predicted_index": int(predictions[index]),
            "predicted_label": predicted_labels[index],
            "predicted_name_zh": CLASS_NAMES_ZH[predicted_labels[index]],
            "classification_margin": f"{margins[index]:.8f}",
        }
        for class_index, class_label in enumerate(CLASS_LABELS):
            row[f"score_{class_label}"] = f"{scores[index, class_index]:.8f}"
        if cube_records:
            row["cube_hdr"] = cube_records[index]["hdr_path"]
            row["cube_raw"] = cube_records[index]["raw_path"]
        rows.append(row)
    write_rows_csv(args.output_dir / "detections_and_classification.csv", rows)

    class_count_rows = [
        {
            "class_index": index,
            "class_label": label,
            "class_name_zh": CLASS_NAMES_ZH[label],
            "count": predicted_labels.count(label),
        }
        for index, label in enumerate(CLASS_LABELS)
    ]
    write_rows_csv(args.output_dir / "class_counts.csv", class_count_rows)

    reference_metrics = None
    if args.reference_pose_label is not None:
        reference_path = args.reference_pose_label.resolve()
        reference_centers = read_reference_centers(reference_path, width, height)
        reference_metrics = evaluate_reference_centers(
            detections,
            predicted_labels,
            reference_centers,
            args.match_radius_px,
            args.reference_class_label,
        )
        reference_metrics["reference_pose_label"] = str(reference_path)
        (args.output_dir / "reference_validation_metrics.json").write_text(
            json.dumps(reference_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[reference] TP={reference_metrics['true_positives']} "
            f"FP={reference_metrics['false_positives']} FN={reference_metrics['false_negatives']} "
            f"F1={reference_metrics['f1']:.4f}"
        )
        if args.reference_class_label is not None:
            matched_accuracy = reference_metrics["matched_classification_accuracy"]
            accuracy_text = f"{matched_accuracy:.4f}" if matched_accuracy is not None else "N/A"
            print(f"[reference class] {args.reference_class_label} accuracy={accuracy_text}")

    if args.geometric_concentration:
        concentration_slope = None
        concentration_intercept = 0.0
        calibration_source = "field_volume_geometry"
    elif args.calibration_slope is not None:
        if args.calibration_slope <= 0:
            raise ValueError("Legacy calibration slope must be positive")
        legacy_intercept = args.calibration_intercept or 0.0
        concentration_slope = 1.0 / args.calibration_slope
        concentration_intercept = -legacy_intercept / args.calibration_slope
        calibration_source = "legacy_user_supplied_N_equals_kC_plus_b"
    else:
        concentration_slope = args.concentration_slope
        concentration_intercept = args.concentration_intercept
        calibration_source = DEFAULT_CONCENTRATION_CALIBRATION

    concentration = concentration_summary(
        predicted_labels,
        args.valid_area_mm2,
        args.drop_volume_ul,
        args.wet_area_mm2,
        concentration_slope,
        concentration_intercept,
        calibration_source,
    )
    (args.output_dir / "concentration_estimate.json").write_text(
        json.dumps(concentration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    final_result_path = args.output_dir / "final_result.png"
    final_result = draw_overlay(
        display_background,
        detections,
        predicted_labels,
        margins,
        args.crop_size,
        concentration,
        final_result_path,
    )
    Image.fromarray(final_result, mode="RGB").save(
        args.output_dir / "detection_classification_overlay.png", quality=95
    )
    draw_contact_sheet(
        display_background,
        detections,
        predicted_labels,
        margins,
        args.crop_size,
        args.output_dir / "roi_contact_sheet.png",
        args.contact_sheet_items,
    )
    show_stage_window(
        "Step 3 - Classification and Concentration",
        final_result,
        args.show_windows,
        args.popup_delay_ms,
    )

    timings["total_seconds"] = time.perf_counter() - started
    summary = {
        "pipeline_status": "complete",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "input_band_count": len(files),
        "input_shape": [height, width],
        "wavelength_range_nm": [args.wavelength_min, args.wavelength_max],
        "detector": {
            "architecture": "YOLO13n-Pose-LSCD-LQE",
            "weights": str(args.detector_weights.resolve()),
            **detector_metadata,
        },
        "classifier": {
            "architecture": "spectral_radial_stats + LinearSVM (pure NumPy inference)",
            "spectral_shift_bands": args.spectral_shift_bands,
            "spectral_shift_warning": (
                "This artifact was trained without class-dependent wavelength alignment."
                if classifier_metadata.get("spectral_calibration") == "class_independent_zero_shift"
                else "Legacy training used class-dependent spectral shifts. The supplied shift is only valid when "
                "obtained from an independent instrument calibration, not from the unknown class label."
            ),
            **classifier_metadata,
            "export": classifier_export_metadata,
        },
        "crop": {
            "size": [args.crop_size, args.crop_size],
            "bands": len(MODEL_WAVELENGTHS),
            "wavelength_range_nm": [float(MODEL_WAVELENGTHS[0]), float(MODEL_WAVELENGTHS[-1])],
            "saved_as_envi": not args.no_save_cubes,
            "envi_data_type": "float32" if not args.no_save_cubes else None,
            "circular_mask": not args.no_circular_mask,
            "spectral_resampling": "linear interpolation to 1 nm before wavelength shift",
            "center_tracking": "fixed pose center across wavelengths",
        },
        "counts": {row["class_label"]: row["count"] for row in class_count_rows},
        "total_count": len(detections),
        "reference_validation": reference_metrics,
        "concentration": concentration,
        "visualization": {
            "interactive_windows": args.show_windows,
            "popup_delay_ms": args.popup_delay_ms,
            "background": f"contrast-enhanced grayscale at {pseudo_metadata['actual_detection_wavelength_nm']:.3f} nm",
            "detection_preview": str(args.output_dir / "detection_preview.png"),
            "final_result": str(final_result_path),
        },
        "timings_seconds": timings,
        "important_limitations": [
            (
                "The concentration uses the final six-gradient 10 um PVC empirical calibration."
                if concentration["mode"] == "empirical_linear_calibration"
                else "The concentration uses a geometric estimate rather than empirical calibration."
            ),
            "Transfer of the PVC count calibration to pathogen spores assumes comparable deposition and detection efficiency.",
            "Classification margins are LinearSVC decision margins, not calibrated probabilities.",
            "The current extractor keeps each pose center fixed across wavelengths; retraining should include "
            "small center jitter or deployment should add per-object spectral tracking when drift is material.",
            "One folder is one hyperspectral field; multiple fields must be processed separately before averaging concentration.",
        ],
    }
    (args.output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[class counts] " + ", ".join(f"{row['class_label']}={row['count']}" for row in class_count_rows))
    print(
        f"[concentration] {concentration['estimated_total_concentration_particles_ml']:.2f} particles/mL "
        f"({concentration['status']})"
    )
    print(f"[done] {args.output_dir} ({timings['total_seconds']:.2f} s)")


if __name__ == "__main__":
    main()
