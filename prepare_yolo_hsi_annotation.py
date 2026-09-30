from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_ORIDATA = ROOT / "oridata"
DEFAULT_OUT = ROOT / "yolo_hsi_detection_dataset"
DEFAULT_CLASSES = ["CMB", "DQB", "DWB", "WQ", "YMXB"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SKIP_FOLDER_PATTERNS = ("WHITE", "BACKGROUND", "BG", "BLANK", "DARK")
METADATA_FIELDS = [
    "sequence_id",
    "class_name",
    "split",
    "source_folder",
    "image_path",
    "label_path",
    "detect_file",
    "detect_wavelength_nm",
    "pca_bands",
    "width",
    "height",
]


@dataclass(frozen=True)
class SequenceRecord:
    class_name: str
    sequence_id: str
    folder: Path
    files: list[Path]
    split: str


@dataclass
class AnnotationState:
    centers: list[tuple[int, int]]
    selected: int = -1
    dirty: bool = False


def parse_pos(path: Path) -> int | None:
    match = re.search(r"pos_(-?\d+)", path.stem)
    if match:
        return int(match.group(1))
    return None


def natural_key(path: Path) -> tuple[int, int | str]:
    pos = parse_pos(path)
    if pos is None:
        return (1, path.name)
    return (0, pos)


def sorted_image_files(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    files.sort(key=natural_key)
    return files


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    return value.strip("._-") or "sequence"


def sequence_sort_key(path: Path) -> tuple[str, int, str]:
    name = path.name
    match = re.search(r"(.+?)[_-]?(\d+)$", name)
    if match:
        return (match.group(1), int(match.group(2)), name)
    return (name, -1, name)


def should_skip_folder(folder: Path) -> bool:
    upper = folder.name.upper()
    return any(upper.startswith(pattern) for pattern in SKIP_FOLDER_PATTERNS)


def infer_class_from_folder(folder: Path, classes: list[str]) -> str | None:
    upper = folder.name.upper()
    for class_name in sorted(classes, key=len, reverse=True):
        class_upper = class_name.upper()
        if upper == class_upper:
            return class_name
        if upper.startswith(class_upper):
            tail = upper[len(class_upper) :]
            if not tail or tail[0].isdigit() or tail[0] in "_- ":
                return class_name
    return None


def deterministic_split(sequence_id: str, val_fraction: float, test_fraction: float, forced_split: str | None) -> str:
    if forced_split:
        return forced_split
    digest = hashlib.sha1(sequence_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def discover_sequences(
    root: Path,
    classes: list[str],
    forced_split: str | None,
    val_fraction: float,
    test_fraction: float,
) -> list[SequenceRecord]:
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    records: list[SequenceRecord] = []
    seen_folders: set[Path] = set()

    direct_files = sorted_image_files(root) if root.exists() else []
    if direct_files:
        class_name = infer_class_from_folder(root, classes) or classes[0]
        sequence_id = sanitize_name(root.name)
        split = deterministic_split(sequence_id, val_fraction, test_fraction, forced_split)
        records.append(SequenceRecord(class_name, sequence_id, root, direct_files, split))
        seen_folders.add(root.resolve())

    for class_name in classes:
        class_dir = root / class_name
        if not class_dir.exists():
            continue

        direct_files = sorted_image_files(class_dir)
        if direct_files:
            sequence_id = sanitize_name(class_name)
            split = deterministic_split(sequence_id, val_fraction, test_fraction, forced_split)
            records.append(SequenceRecord(class_name, sequence_id, class_dir, direct_files, split))
            seen_folders.add(class_dir.resolve())
            continue

        for subdir in sorted([p for p in class_dir.iterdir() if p.is_dir()], key=sequence_sort_key):
            files = sorted_image_files(subdir)
            if not files:
                continue
            sequence_id = sanitize_name(f"{class_name}_{subdir.name}")
            split = deterministic_split(sequence_id, val_fraction, test_fraction, forced_split)
            records.append(SequenceRecord(class_name, sequence_id, subdir, files, split))
            seen_folders.add(subdir.resolve())

    for subdir in sorted([p for p in root.iterdir() if p.is_dir()], key=sequence_sort_key):
        if subdir.resolve() in seen_folders or should_skip_folder(subdir):
            continue
        class_name = infer_class_from_folder(subdir, classes)
        if class_name is None:
            continue
        files = sorted_image_files(subdir)
        if not files:
            continue
        sequence_id = sanitize_name(subdir.name)
        split = deterministic_split(sequence_id, val_fraction, test_fraction, forced_split)
        records.append(SequenceRecord(class_name, sequence_id, subdir, files, split))
        seen_folders.add(subdir.resolve())

    records.sort(key=lambda item: (classes.index(item.class_name), sequence_sort_key(Path(item.sequence_id))))
    return records


def label_has_boxes(label_path: Path) -> bool:
    return label_path.exists() and bool(label_path.read_text(encoding="utf-8").strip())


def print_records_summary(records: list[SequenceRecord], out_root: Path) -> None:
    print(f"[records] {len(records)} sequences")
    counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for record in records:
        counts[record.class_name] = counts.get(record.class_name, 0) + 1
        split_key = f"{record.class_name}/{record.split}"
        split_counts[split_key] = split_counts.get(split_key, 0) + 1
    print("[by class]")
    for class_name, count in sorted(counts.items()):
        print(f"  {class_name}: {count}")
    print("[by class/split]")
    for key, count in sorted(split_counts.items()):
        print(f"  {key}: {count}")
    print("[first 20]")
    for index, record in enumerate(records[:20], start=1):
        label_path = yolo_label_path(out_root, record.split, record.sequence_id)
        status = "labeled" if label_has_boxes(label_path) else "unlabeled"
        print(f"  {index:03d}. {record.sequence_id:16s} class={record.class_name:5s} split={record.split:5s} files={len(record.files):3d} {status}")


def wavelength_axis(files: list[Path], wavelength_min: float, wavelength_max: float) -> np.ndarray:
    return np.linspace(wavelength_min, wavelength_max, len(files), dtype=np.float64)


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def robust_norm(image: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.uint8)
    low = float(np.percentile(finite, low_pct))
    high = float(np.percentile(finite, high_pct))
    if high <= low:
        low = float(finite.min())
        high = float(finite.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    out = np.clip((image - low) / (high - low), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def normalize_float(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    low = float(np.percentile(values, 1.0))
    high = float(np.percentile(values, 99.5))
    if high <= low:
        low = float(values.min())
        high = float(values.max())
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def clahe_channel(image_u8: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image_u8)


def select_band_file(files: list[Path], wavelength_nm: float, wavelength_min: float, wavelength_max: float) -> tuple[Path, float]:
    wavelengths = wavelength_axis(files, wavelength_min, wavelength_max)
    index = int(np.argmin(np.abs(wavelengths - wavelength_nm)))
    return files[index], float(wavelengths[index])


def choose_pca_files(files: list[Path], pca_bands: int) -> list[Path]:
    if pca_bands <= 1 or pca_bands >= len(files):
        return files
    indices = np.linspace(0, len(files) - 1, pca_bands).round().astype(int)
    indices = np.unique(indices)
    return [files[int(i)] for i in indices]


def pca_first_component(files: list[Path], sample_pixels: int, rng: np.random.Generator) -> np.ndarray:
    stack = np.stack([read_gray(path) for path in files], axis=0).astype(np.float32)
    bands, height, width = stack.shape
    flattened = stack.reshape(bands, -1)
    total_pixels = flattened.shape[1]
    if sample_pixels > 0 and sample_pixels < total_pixels:
        sample_idx = rng.choice(total_pixels, size=sample_pixels, replace=False)
        sample = flattened[:, sample_idx]
    else:
        sample = flattened
    means = sample.mean(axis=1, keepdims=True)
    stds = sample.std(axis=1, keepdims=True) + 1e-6
    sample_z = (sample - means) / stds
    cov = (sample_z @ sample_z.T) / max(1, sample_z.shape[1] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    weights = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
    if weights.sum() < 0:
        weights *= -1.0
    full_z = (flattened - means) / stds
    pc1 = weights @ full_z
    return pc1.reshape(height, width).astype(np.float32)


def fringe_energy(image: np.ndarray) -> np.ndarray:
    image_u8 = robust_norm(image)
    enhanced = clahe_channel(image_u8).astype(np.float32)
    background = cv2.GaussianBlur(enhanced, ksize=(0, 0), sigmaX=18.0, sigmaY=18.0)
    highpass = enhanced - background
    lap = cv2.Laplacian(cv2.GaussianBlur(enhanced, ksize=(0, 0), sigmaX=1.2), cv2.CV_32F, ksize=3)
    return 0.65 * np.abs(highpass) + 0.35 * np.abs(lap)


def build_pseudo_rgb(
    files: list[Path],
    detect_wavelength: float,
    wavelength_min: float,
    wavelength_max: float,
    pca_bands: int,
    pca_sample_pixels: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, str | float | int]]:
    band_file, actual_wavelength = select_band_file(files, detect_wavelength, wavelength_min, wavelength_max)
    band_image = read_gray(band_file)
    pca_files = choose_pca_files(files, pca_bands)
    rng = np.random.default_rng(seed)
    pc1 = pca_first_component(pca_files, sample_pixels=pca_sample_pixels, rng=rng)
    fringe = fringe_energy(band_image)

    red = clahe_channel(robust_norm(band_image))
    green = clahe_channel(robust_norm(pc1, low_pct=0.5, high_pct=99.5))
    blue = clahe_channel(robust_norm(fringe, low_pct=1.0, high_pct=99.7))
    rgb = np.dstack([red, green, blue]).astype(np.uint8)
    meta = {
        "detect_file": band_file.name,
        "detect_wavelength_nm": actual_wavelength,
        "pca_bands": len(pca_files),
        "height": int(rgb.shape[0]),
        "width": int(rgb.shape[1]),
    }
    return rgb, meta


def yolo_label_path(out_root: Path, split: str, sequence_id: str) -> Path:
    return out_root / "labels" / split / f"{sequence_id}.txt"


def yolo_image_path(out_root: Path, split: str, sequence_id: str) -> Path:
    return out_root / "images" / split / f"{sequence_id}.png"


def load_centers_from_yolo(label_path: Path, width: int, height: int) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    if not label_path.exists():
        return centers
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
        except ValueError:
            continue
        centers.append((int(round(x_center)), int(round(y_center))))
    return centers


def write_yolo_labels(
    label_path: Path,
    centers: list[tuple[int, int]],
    image_width: int,
    image_height: int,
    box_size: int,
    class_id: int,
) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    box_w = box_size / image_width
    box_h = box_size / image_height
    lines = []
    for x_center, y_center in centers:
        x_norm = np.clip(x_center / image_width, 0.0, 1.0)
        y_norm = np.clip(y_center / image_height, 0.0, 1.0)
        lines.append(f"{class_id} {x_norm:.8f} {y_norm:.8f} {box_w:.8f} {box_h:.8f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def prelabel_response(rgb: np.ndarray) -> np.ndarray:
    red = rgb[:, :, 0].astype(np.float32)
    green = rgb[:, :, 1].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)
    gray = np.clip(0.45 * red + 0.25 * green + 0.30 * blue, 0, 255).astype(np.uint8)
    enhanced = clahe_channel(gray)
    smooth = cv2.GaussianBlur(enhanced, ksize=(0, 0), sigmaX=1.2, sigmaY=1.2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    white = cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    black = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
    local_blob = np.maximum(white, black)
    background = cv2.GaussianBlur(smooth.astype(np.float32), ksize=(0, 0), sigmaX=18.0, sigmaY=18.0)
    highpass = np.abs(smooth.astype(np.float32) - background)
    lap = np.abs(cv2.Laplacian(cv2.GaussianBlur(smooth, ksize=(0, 0), sigmaX=1.0), cv2.CV_32F, ksize=3))
    response = (
        0.36 * normalize_float(local_blob)
        + 0.26 * normalize_float(highpass)
        + 0.20 * normalize_float(lap)
        + 0.18 * normalize_float(blue)
    )
    return cv2.GaussianBlur(response, ksize=(0, 0), sigmaX=1.0, sigmaY=1.0).astype(np.float32)


def radial_support_from_rgb(rgb: np.ndarray, x_center: int, y_center: int, box_size: int) -> float:
    half = box_size // 2
    patch = rgb[y_center - half : y_center + half, x_center - half : x_center + half]
    if patch.shape[:2] != (box_size, box_size):
        return 0.0
    red = patch[:, :, 0].astype(np.float32)
    blue = patch[:, :, 2].astype(np.float32)
    signal = 0.45 * red + 0.55 * blue
    background = cv2.GaussianBlur(signal, ksize=(0, 0), sigmaX=9.0, sigmaY=9.0)
    energy = np.abs(signal - background)

    yy, xx = np.mgrid[:box_size, :box_size]
    cy = (box_size - 1) / 2.0
    cx = (box_size - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    theta = np.arctan2(yy - cy, xx - cx)
    inner = max(7.0, box_size * 0.09)
    outer = min(half - 7.0, box_size * 0.43)
    annulus = (radius >= inner) & (radius <= outer)
    outer_ring = (radius >= outer + 1.0) & (radius <= half - 2.0)
    if not annulus.any() or not outer_ring.any():
        return 0.0

    radius_i = radius.astype(int)
    profile = []
    for rad in range(int(math.ceil(inner)), int(math.floor(outer)) + 1):
        mask = radius_i == rad
        if mask.any():
            profile.append(float(energy[mask].mean()))
    if len(profile) < 5:
        return 0.0
    profile_arr = np.asarray(profile, dtype=np.float32)
    profile_mean = float(profile_arr.mean())
    if profile_mean <= 1e-6:
        return 0.0

    sector_values = []
    for sector in range(24):
        lo = -math.pi + sector * 2.0 * math.pi / 24.0
        hi = -math.pi + (sector + 1) * 2.0 * math.pi / 24.0
        mask = annulus & (theta >= lo) & (theta < hi)
        if mask.any():
            sector_values.append(float(energy[mask].mean()))
    if len(sector_values) < 8:
        return 0.0
    sector_arr = np.asarray(sector_values, dtype=np.float32)

    radial_oscillation = float(profile_arr.std() / (profile_mean + 1e-6))
    angular_balance = float(np.percentile(sector_arr, 25) / (np.percentile(sector_arr, 75) + 1e-6))
    ring_peak = float(np.percentile(profile_arr, 90) / (np.percentile(profile_arr, 35) + 1e-6))
    annulus_ratio = float(energy[annulus].mean() / (energy[outer_ring].mean() + 1e-6))
    support = radial_oscillation * angular_balance * ring_peak * min(max(annulus_ratio, 0.0), 2.5)
    if not math.isfinite(support):
        return 0.0
    return float(max(support, 0.0))


def nonmax_centers(candidates: list[tuple[int, int, float]], min_distance: int, max_detections: int) -> list[tuple[int, int]]:
    kept: list[tuple[int, int, float]] = []
    min_dist_sq = float(min_distance * min_distance)
    for x_center, y_center, score in sorted(candidates, key=lambda item: item[2], reverse=True):
        if all((x_center - x_prev) ** 2 + (y_center - y_prev) ** 2 >= min_dist_sq for x_prev, y_prev, _ in kept):
            kept.append((x_center, y_center, score))
        if len(kept) >= max_detections:
            break
    return [(x_center, y_center) for x_center, y_center, _ in kept]


def auto_detect_centers(
    rgb: np.ndarray,
    box_size: int,
    max_detections: int,
    min_distance: int,
    threshold_percentile: float,
    min_ring_support: float,
) -> list[tuple[int, int]]:
    response = prelabel_response(rgb)
    height, width = response.shape
    half = box_size // 2
    threshold = float(np.percentile(response, threshold_percentile))
    window = max(9, int(min_distance // 2) * 2 + 1)
    dilated = cv2.dilate(response, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (window, window)))
    mask = (response >= threshold) & (response >= dilated - 1e-6)
    mask[: half + 2, :] = False
    mask[height - half - 2 :, :] = False
    mask[:, : half + 2] = False
    mask[:, width - half - 2 :] = False
    rows, cols = np.nonzero(mask)

    candidates: list[tuple[int, int, float]] = []
    for row, col in zip(rows.tolist(), cols.tolist()):
        support = radial_support_from_rgb(rgb, int(col), int(row), box_size)
        if support < min_ring_support:
            continue
        score = float(response[row, col] * (0.25 + min(support, 3.0)))
        candidates.append((int(col), int(row), score))
    return nonmax_centers(candidates, min_distance=min_distance, max_detections=max_detections)


def draw_annotation_view(
    image: np.ndarray,
    state: AnnotationState,
    box_size: int,
    scale: float,
    title: str,
) -> np.ndarray:
    view = cv2.resize(image, (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    half = box_size / 2.0
    for idx, (x_center, y_center) in enumerate(state.centers):
        color = (0, 255, 255) if idx != state.selected else (0, 80, 255)
        x0 = int(round((x_center - half) * scale))
        y0 = int(round((y_center - half) * scale))
        x1 = int(round((x_center + half) * scale))
        y1 = int(round((y_center + half) * scale))
        cv2.rectangle(view, (x0, y0), (x1, y1), color, 2)
        cv2.circle(view, (int(round(x_center * scale)), int(round(y_center * scale))), 3, color, -1)
        cv2.putText(view, str(idx + 1), (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    help_lines = [
        title,
        f"boxes={len(state.centers)} fixed={box_size}px | left add/select | right delete nearest",
        "s save next | b previous | u undo | c clear | q quit",
    ]
    y = 22
    for line in help_lines:
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return view


def nearest_center(centers: list[tuple[int, int]], x: int, y: int, max_distance: float) -> int:
    if not centers:
        return -1
    distances = [math.hypot(cx - x, cy - y) for cx, cy in centers]
    index = int(np.argmin(distances))
    return index if distances[index] <= max_distance else -1


def annotate_image(
    image: np.ndarray,
    label_path: Path,
    box_size: int,
    class_id: int,
    max_display_width: int,
    max_display_height: int,
    title: str,
) -> str:
    height, width = image.shape[:2]
    scale = min(max_display_width / width, max_display_height / height, 1.0)
    state = AnnotationState(centers=load_centers_from_yolo(label_path, width, height))
    history: list[list[tuple[int, int]]] = []
    window_name = "HSI YOLO fixed-box annotator"

    def push_history() -> None:
        history.append(list(state.centers))
        if len(history) > 50:
            history.pop(0)

    def mouse_callback(event: int, x_view: int, y_view: int, _flags: int, _param: object) -> None:
        x = int(round(x_view / scale))
        y = int(round(y_view / scale))
        x = int(np.clip(x, box_size // 2, width - box_size // 2))
        y = int(np.clip(y, box_size // 2, height - box_size // 2))
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = nearest_center(state.centers, x, y, max_distance=box_size * 0.45)
            push_history()
            if idx >= 0:
                state.centers[idx] = (x, y)
                state.selected = idx
            else:
                state.centers.append((x, y))
                state.selected = len(state.centers) - 1
            state.dirty = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = nearest_center(state.centers, x, y, max_distance=box_size)
            if idx >= 0:
                push_history()
                state.centers.pop(idx)
                state.selected = -1
                state.dirty = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    while True:
        view = draw_annotation_view(image, state, box_size, scale, title)
        cv2.imshow(window_name, view)
        key = cv2.waitKey(30) & 0xFF
        if key in (255, 0xFF):
            continue
        if key == ord("s"):
            write_yolo_labels(label_path, state.centers, width, height, box_size, class_id)
            state.dirty = False
            return "next"
        if key == ord("b"):
            write_yolo_labels(label_path, state.centers, width, height, box_size, class_id)
            state.dirty = False
            return "previous"
        if key == ord("u") and history:
            state.centers = history.pop()
            state.selected = -1
            state.dirty = True
        if key == ord("c"):
            push_history()
            state.centers.clear()
            state.selected = -1
            state.dirty = True
        if key == ord("q"):
            if state.dirty:
                write_yolo_labels(label_path, state.centers, width, height, box_size, class_id)
            return "quit"


def ensure_dataset_dirs(out_root: Path) -> None:
    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_files(out_root: Path, label_names: list[str]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "classes.txt").write_text("\n".join(label_names) + "\n", encoding="utf-8")
    names = ", ".join([f"'{name}'" for name in label_names])
    yaml_text = (
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(label_names)}\n"
        f"names: [{names}]\n"
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def write_metadata_header(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(METADATA_FIELDS)


def upsert_metadata(path: Path, record: SequenceRecord, image_path: Path, label_path: Path, meta: dict[str, str | float | int]) -> None:
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("sequence_id") != record.sequence_id:
                    rows.append({field: row.get(field, "") for field in METADATA_FIELDS})
    rows.append(
        {
            "sequence_id": record.sequence_id,
            "class_name": record.class_name,
            "split": record.split,
            "source_folder": str(record.folder),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "detect_file": str(meta["detect_file"]),
            "detect_wavelength_nm": str(meta["detect_wavelength_nm"]),
            "pca_bands": str(meta["pca_bands"]),
            "width": str(meta["width"]),
            "height": str(meta["height"]),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path)


def label_names_for_mode(label_mode: str, classes: list[str]) -> list[str]:
    if label_mode == "class":
        return classes
    return ["particle"]


def class_id_for_record(record: SequenceRecord, label_mode: str, classes: list[str]) -> int:
    if label_mode == "class":
        return classes.index(record.class_name)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create pseudo-RGB hyperspectral detection images and annotate fixed-size YOLO boxes by clicking diffraction centers."
    )
    parser.add_argument("--oridata-root", type=Path, default=DEFAULT_ORIDATA)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES)
    parser.add_argument("--label-mode", choices=["particle", "class"], default="particle")
    parser.add_argument("--split", choices=["train", "val", "test"], default=None, help="Force all generated images into one split.")
    parser.add_argument("--val-fraction", type=float, default=0.20, help="Used only when --split is omitted.")
    parser.add_argument("--test-fraction", type=float, default=0.00, help="Used only when --split is omitted.")
    parser.add_argument("--box-size", type=int, default=100, help="Fixed YOLO box size in original image pixels.")
    parser.add_argument("--detect-wavelength", type=float, default=480.0)
    parser.add_argument("--wavelength-min", type=float, default=400.0)
    parser.add_argument("--wavelength-max", type=float, default=800.0)
    parser.add_argument("--pca-bands", type=int, default=24, help="Evenly sampled bands used to compute the pseudo-RGB PCA channel.")
    parser.add_argument("--pca-sample-pixels", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-display-width", type=int, default=1500)
    parser.add_argument("--max-display-height", type=int, default=950)
    parser.add_argument("--prepare-only", action="store_true", help="Only generate pseudo-RGB images and dataset files; do not open the annotator.")
    parser.add_argument("--auto-prelabel", action="store_true", help="Before annotation, write fixed-box candidate labels when the label file is empty.")
    parser.add_argument("--prelabel-only", action="store_true", help="Generate pseudo-RGB images and automatic candidate labels, then exit without opening the annotator.")
    parser.add_argument("--overwrite-labels", action="store_true", help="Allow auto-prelabel to replace existing YOLO labels.")
    parser.add_argument("--prelabel-max-detections", type=int, default=260)
    parser.add_argument("--prelabel-min-distance", type=int, default=42)
    parser.add_argument("--prelabel-threshold-percentile", type=float, default=99.15)
    parser.add_argument("--prelabel-min-ring-support", type=float, default=0.08)
    parser.add_argument("--overwrite-images", action="store_true")
    parser.add_argument("--list-only", action="store_true", help="Only list discovered sequences and inferred classes.")
    parser.add_argument("--skip-labeled", action="store_true", help="When annotating, skip images whose YOLO label file already contains boxes.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-sequences", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.prelabel_only:
        args.auto_prelabel = True
        args.prepare_only = True
    classes = list(args.classes)
    records = discover_sequences(args.oridata_root, classes, args.split, args.val_fraction, args.test_fraction)
    if args.max_sequences is not None:
        records = records[args.start_index : args.start_index + args.max_sequences]
    else:
        records = records[args.start_index :]
    if not records:
        raise SystemExit(f"No image sequences found under {args.oridata_root}")
    if args.list_only:
        print_records_summary(records, args.out_root)
        return

    ensure_dataset_dirs(args.out_root)
    write_dataset_files(args.out_root, label_names_for_mode(args.label_mode, classes))
    metadata_path = args.out_root / "metadata.csv"
    print(f"[dataset] {args.out_root}")
    print(f"[records] {len(records)}")

    index = 0
    while 0 <= index < len(records):
        record = records[index]
        image_path = yolo_image_path(args.out_root, record.split, record.sequence_id)
        label_path = yolo_label_path(args.out_root, record.split, record.sequence_id)
        if args.skip_labeled and label_has_boxes(label_path):
            print(f"[skip labeled] {record.sequence_id}: {label_path}")
            index += 1
            continue
        if image_path.exists() and not args.overwrite_images:
            rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            meta = {
                "detect_file": "",
                "detect_wavelength_nm": args.detect_wavelength,
                "pca_bands": args.pca_bands,
                "height": rgb.shape[0],
                "width": rgb.shape[1],
            }
            print(f"[reuse] {record.sequence_id}: {image_path}")
            if not metadata_path.exists():
                upsert_metadata(metadata_path, record, image_path, label_path, meta)
        else:
            print(f"[build] {index + 1}/{len(records)} {record.sequence_id} ({record.class_name}, {record.split})")
            rgb, meta = build_pseudo_rgb(
                record.files,
                detect_wavelength=args.detect_wavelength,
                wavelength_min=args.wavelength_min,
                wavelength_max=args.wavelength_max,
                pca_bands=args.pca_bands,
                pca_sample_pixels=args.pca_sample_pixels,
                seed=args.seed + index,
            )
            save_rgb(image_path, rgb)
            upsert_metadata(metadata_path, record, image_path, label_path, meta)

        if args.auto_prelabel and (args.overwrite_labels or not label_has_boxes(label_path)):
            centers = auto_detect_centers(
                rgb,
                box_size=args.box_size,
                max_detections=args.prelabel_max_detections,
                min_distance=args.prelabel_min_distance,
                threshold_percentile=args.prelabel_threshold_percentile,
                min_ring_support=args.prelabel_min_ring_support,
            )
            write_yolo_labels(
                label_path,
                centers,
                image_width=rgb.shape[1],
                image_height=rgb.shape[0],
                box_size=args.box_size,
                class_id=class_id_for_record(record, args.label_mode, classes),
            )
            print(f"[prelabel] {record.sequence_id}: {len(centers)} boxes -> {label_path}")

        if args.prepare_only:
            if not label_path.exists():
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("", encoding="utf-8")
            index += 1
            continue

        title = f"{index + 1}/{len(records)} {record.sequence_id} split={record.split}"
        action = annotate_image(
            rgb,
            label_path,
            box_size=args.box_size,
            class_id=class_id_for_record(record, args.label_mode, classes),
            max_display_width=args.max_display_width,
            max_display_height=args.max_display_height,
            title=title,
        )
        if action == "quit":
            break
        if action == "previous":
            index = max(0, index - 1)
        else:
            index += 1

    cv2.destroyAllWindows()
    print(f"[done] images: {args.out_root / 'images'}")
    print(f"[done] labels: {args.out_root / 'labels'}")
    print(f"[done] yaml: {args.out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
