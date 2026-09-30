from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
MODEL_WAVELENGTHS = np.arange(411.0, 780.0, 1.0, dtype=np.float64)
CLASS_SPECS = [
    ("DATACMB", "cmb", "CMB"),
    ("DATADQB", "dqb", "DQB"),
    ("DATADWB", "dwb", "DWB"),
    ("DATAWQ", "wq", "WQ"),
    ("DATAYMXB", "ymxb", "YMXB"),
]
DTYPE_MAP = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


@dataclass(frozen=True)
class Sample:
    hdr_path: Path
    raw_path: Path
    label_index: int
    label: str
    split: str


def numeric_suffix(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def discover_samples(root: Path, train_per_class: int, test_per_class: int) -> list[Sample]:
    samples: list[Sample] = []
    required = train_per_class + test_per_class
    for label_index, (folder_name, _prefix, label) in enumerate(CLASS_SPECS):
        headers = sorted((root / folder_name).glob("*.hdr"), key=numeric_suffix)[:required]
        if len(headers) != required:
            raise ValueError(f"{folder_name}: expected {required} cubes, found {len(headers)}")
        for index, hdr_path in enumerate(headers):
            raw_path = hdr_path.with_suffix(".raw")
            if not raw_path.exists():
                raise FileNotFoundError(raw_path)
            samples.append(
                Sample(
                    hdr_path=hdr_path,
                    raw_path=raw_path,
                    label_index=label_index,
                    label=label,
                    split="train" if index < train_per_class else "test",
                )
            )
    return samples


def parse_header_value(text: str, field: str) -> int:
    match = re.search(rf"^{re.escape(field)}\s*=\s*(\d+)", text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        raise ValueError(f"Missing ENVI field: {field}")
    return int(match.group(1))


def load_resampled_cube(sample: Sample) -> np.ndarray:
    header = sample.hdr_path.read_text(encoding="utf-8", errors="ignore")
    bands = parse_header_value(header, "bands")
    lines = parse_header_value(header, "lines")
    samples = parse_header_value(header, "samples")
    data_type = parse_header_value(header, "data type")
    dtype = DTYPE_MAP.get(data_type)
    if dtype is None:
        raise ValueError(f"Unsupported ENVI data type {data_type}: {sample.hdr_path}")
    wavelength_match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", header, flags=re.IGNORECASE | re.DOTALL)
    wavelength_values = (
        [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", wavelength_match.group(1))]
        if wavelength_match
        else []
    )
    raw_wavelengths = (
        np.asarray(wavelength_values, dtype=np.float64)
        if len(wavelength_values) == bands
        else np.linspace(400.0, 800.0, bands, dtype=np.float64)
    )
    expected = bands * lines * samples
    raw = np.fromfile(sample.raw_path, dtype=dtype, count=expected)
    if raw.size != expected:
        raise ValueError(f"Unexpected raw cube size: {sample.raw_path}")
    raw = raw.reshape(bands, lines, samples)
    positions = np.interp(MODEL_WAVELENGTHS, raw_wavelengths, np.arange(bands, dtype=np.float64))
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    weights = (positions - lower).astype(np.float32)[:, None, None]
    cube = np.asarray(raw[lower], dtype=np.float32) * (1.0 - weights)
    cube += np.asarray(raw[upper], dtype=np.float32) * weights
    if cube.shape[1:] == (100, 100):
        return cube
    fitted = np.zeros((len(MODEL_WAVELENGTHS), 100, 100), dtype=np.float32)
    copy_height = min(cube.shape[1], 100)
    copy_width = min(cube.shape[2], 100)
    source_y = max((cube.shape[1] - 100) // 2, 0)
    source_x = max((cube.shape[2] - 100) // 2, 0)
    target_y = max((100 - cube.shape[1]) // 2, 0)
    target_x = max((100 - cube.shape[2]) // 2, 0)
    fitted[:, target_y : target_y + copy_height, target_x : target_x + copy_width] = cube[
        :, source_y : source_y + copy_height, source_x : source_x + copy_width
    ]
    return fitted


def mean_nonzero_spectrum(cube: np.ndarray) -> np.ndarray:
    flat = cube.reshape(cube.shape[0], -1)
    valid = flat > 0
    counts = valid.sum(axis=1)
    sums = np.where(valid, flat, 0.0).sum(axis=1)
    fallback = flat.mean(axis=1)
    return np.divide(sums, counts, out=fallback, where=counts > 0).astype(np.float32)


def select_wavelength_indices(
    train_samples: list[Sample],
    top_k: int,
    min_gap_nm: float,
) -> np.ndarray:
    spectra: list[np.ndarray] = []
    labels: list[int] = []
    for index, sample in enumerate(train_samples, start=1):
        spectra.append(mean_nonzero_spectrum(load_resampled_cube(sample)))
        labels.append(sample.label_index)
        if index % 50 == 0 or index == len(train_samples):
            print(f"[feature selection] {index}/{len(train_samples)}")
    f_values, _ = f_classif(np.vstack(spectra), np.asarray(labels, dtype=int))
    order = np.argsort(np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0))[::-1]
    selected: list[int] = []
    for candidate in order:
        wavelength = MODEL_WAVELENGTHS[candidate]
        if all(abs(wavelength - MODEL_WAVELENGTHS[chosen]) >= min_gap_nm for chosen in selected):
            selected.append(int(candidate))
        if len(selected) == top_k:
            break
    if len(selected) != top_k:
        raise RuntimeError(f"Only {len(selected)} wavelengths satisfy the requested spacing")
    return np.asarray(selected, dtype=int)


def radial_geometry(bins: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:100, :100]
    radius = np.sqrt((yy - 49.5) ** 2 + (xx - 49.5) ** 2)
    ids = np.floor(radius / (radius.max() + 1e-6) * bins).astype(np.int64)
    ids = np.clip(ids, 0, bins - 1)
    counts = np.bincount(ids.reshape(-1), minlength=bins).astype(np.float32)
    counts[counts == 0] = 1.0
    return ids, counts


def feature_from_cube(
    cube: np.ndarray,
    selected_indices: np.ndarray,
    bin_ids: np.ndarray,
    bin_counts: np.ndarray,
) -> np.ndarray:
    spectrum = mean_nonzero_spectrum(cube)
    selected = spectrum[selected_indices]
    selected_normalized = (selected - selected.mean()) / (selected.std() + 1e-6)
    radial_features: list[np.ndarray] = []
    for band_index in selected_indices:
        profile = np.bincount(
            bin_ids.reshape(-1),
            weights=cube[band_index].reshape(-1),
            minlength=len(bin_counts),
        )[: len(bin_counts)]
        profile = profile / bin_counts
        maximum = float(np.max(np.abs(profile)))
        if maximum > 0:
            profile = profile / maximum
        radial_features.append(profile.astype(np.float32))
    global_features = np.asarray(
        [
            spectrum.mean(),
            spectrum.std(),
            np.percentile(spectrum, 10),
            np.percentile(spectrum, 90),
            cube.mean(),
            cube.std(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([selected_normalized, *radial_features, global_features]).astype(np.float32)


def extract_features(
    samples: list[Sample],
    selected_indices: np.ndarray,
    radial_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    bin_ids, bin_counts = radial_geometry(radial_bins)
    features: list[np.ndarray] = []
    labels: list[int] = []
    for index, sample in enumerate(samples, start=1):
        features.append(feature_from_cube(load_resampled_cube(sample), selected_indices, bin_ids, bin_counts))
        labels.append(sample.label_index)
        if index % 50 == 0 or index == len(samples):
            print(f"[features] {samples[0].split} {index}/{len(samples)}")
    return np.vstack(features).astype(np.float32), np.asarray(labels, dtype=int)


def export_numpy_model(
    model,
    output_path: Path,
    labels: list[str],
    selected_wavelengths: np.ndarray,
    radial_bins: int,
) -> None:
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["linearsvc"]
    np.savez_compressed(
        output_path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coefficients=classifier.coef_.astype(np.float32),
        intercept=classifier.intercept_.astype(np.float32),
        classes=classifier.classes_.astype(np.int64),
        labels=np.asarray(labels),
        selected_wavelengths=selected_wavelengths.astype(np.float32),
        radial_bins=np.asarray([radial_bins], dtype=np.int64),
        input_variant=np.asarray(["spectral_radial_stats"]),
        spectral_calibration=np.asarray(["class_independent_zero_shift"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a class-independent zero-shift Jetson classifier.")
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "deployment_classifier_class_independent",
    )
    parser.add_argument("--train-per-class", type=int, default=70)
    parser.add_argument("--test-per-class", type=int, default=30)
    parser.add_argument("--top-wavelengths", type=int, default=40)
    parser.add_argument("--min-wavelength-gap-nm", type=float, default=5.0)
    parser.add_argument("--radial-bins", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = discover_samples(args.data_root.resolve(), args.train_per_class, args.test_per_class)
    train_samples = [sample for sample in samples if sample.split == "train"]
    test_samples = [sample for sample in samples if sample.split == "test"]
    selected_indices = select_wavelength_indices(
        train_samples,
        args.top_wavelengths,
        args.min_wavelength_gap_nm,
    )
    selected_wavelengths = MODEL_WAVELENGTHS[selected_indices]
    print("[selected] " + ", ".join(f"{value:.0f}" for value in selected_wavelengths))
    train_x, train_y = extract_features(train_samples, selected_indices, args.radial_bins)
    test_x, test_y = extract_features(test_samples, selected_indices, args.radial_bins)

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual=False,
            max_iter=20000,
            random_state=args.seed,
        ),
    )
    model.fit(train_x, train_y)
    predictions = model.predict(test_x).astype(int)
    labels = [spec[2] for spec in CLASS_SPECS]
    metrics = {
        "training_preprocessing": "class-independent zero-shift 1-nm linear resampling",
        "train_n": int(len(train_y)),
        "test_n": int(len(test_y)),
        "feature_dimension": int(train_x.shape[1]),
        "accuracy": float(accuracy_score(test_y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        "macro_f1": float(f1_score(test_y, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(test_y, predictions).tolist(),
        "classification_report": classification_report(
            test_y,
            predictions,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
    }

    pickle_path = args.output_dir / "spectral_radial_stats_linear_svm_class_independent.pkl"
    with pickle_path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    numpy_path = args.output_dir / "spectral_radial_stats_linear_svm_class_independent.npz"
    export_numpy_model(model, numpy_path, labels, selected_wavelengths, args.radial_bins)
    with (args.output_dir / "selected_wavelengths.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "band_index", "wavelength_nm"])
        writer.writeheader()
        for rank, (index, wavelength) in enumerate(zip(selected_indices, selected_wavelengths), start=1):
            writer.writerow({"rank": rank, "band_index": int(index), "wavelength_nm": float(wavelength)})
    with (args.output_dir / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "true_label", "predicted_label", "correct"])
        writer.writeheader()
        for sample, true_index, predicted_index in zip(test_samples, test_y, predictions):
            writer.writerow(
                {
                    "sample": sample.hdr_path.stem,
                    "true_label": labels[int(true_index)],
                    "predicted_label": labels[int(predicted_index)],
                    "correct": int(true_index == predicted_index),
                }
            )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[result] accuracy={metrics['accuracy']:.4f} balanced_accuracy={metrics['balanced_accuracy']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print(f"[saved] {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
