from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from pinn_diffusion_hsi_augmentation import (
    CATEGORY_SPECS,
    EnviMeta,
    SampleRecord,
    choose_template_meta,
    fit_cube_to_template,
    natural_key,
    read_envi_cube,
    read_envi_meta,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "ENVI_CORRECTED_CROPPED"
DEFAULT_PINN_DIR = ROOT / "ENVI_PINN_DIFFUSION_AUGMENTED"
DEFAULT_DDPM_DIR = ROOT / "ENVI_DDPM_AUGMENTED"
DEFAULT_OUT_ROOT = ROOT / "augmentation_classification_results"


@dataclass(frozen=True)
class FeatureBundle:
    names: list[str]
    labels: list[str]
    y: np.ndarray
    x: np.ndarray


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def discover_real_records(data_root: Path, test: bool) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    suffix = "_test" if test else ""
    for class_index, (prefix, folder, label) in enumerate(CATEGORY_SPECS):
        folder_path = data_root / f"{folder}{suffix}"
        if not folder_path.exists():
            raise FileNotFoundError(f"Missing expected folder: {folder_path}")
        hdrs = sorted(folder_path.glob("*.hdr"), key=natural_key)
        for hdr in hdrs:
            records.append(
                SampleRecord(
                    name=f"{prefix}_{hdr.stem}",
                    prefix=prefix,
                    folder=folder_path.name,
                    label=label,
                    class_index=class_index,
                    hdr_path=hdr,
                )
            )
    return records


def discover_augmented_records(root: Path, method_label: str, per_class: int | None) -> list[SampleRecord]:
    if not root.exists():
        print(f"[augment] {method_label}: missing directory, skipped: {root}")
        return []
    records: list[SampleRecord] = []
    for class_index, (prefix, folder, label) in enumerate(CATEGORY_SPECS):
        folder_path = root / folder
        if not folder_path.exists():
            print(f"[augment] {method_label}: missing class folder, skipped: {folder_path}")
            continue
        hdrs = sorted(folder_path.glob("*.hdr"), key=natural_key)
        if per_class is not None and per_class > 0:
            hdrs = hdrs[:per_class]
        for hdr in hdrs:
            records.append(
                SampleRecord(
                    name=f"{method_label}_{prefix}_{hdr.stem}",
                    prefix=prefix,
                    folder=folder_path.name,
                    label=label,
                    class_index=class_index,
                    hdr_path=hdr,
                )
            )
    print(f"[augment] {method_label}: loaded {len(records)} records from {root}")
    return records


def summarize_records(records: Iterable[SampleRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec.label] = counts.get(rec.label, 0) + 1
    return counts


def read_fitted_cube(path: Path, template: EnviMeta) -> np.ndarray:
    meta, cube = read_envi_cube(path)
    if meta.bands != template.bands:
        raise ValueError(f"Band mismatch at {path}: {meta.bands} != {template.bands}")
    if not np.allclose(meta.wavelengths, template.wavelengths):
        raise ValueError(f"Wavelength mismatch at {path}")
    return fit_cube_to_template(cube, template)


def mean_nonzero_spectrum(cube: np.ndarray) -> np.ndarray:
    flat = cube.reshape(cube.shape[0], -1).astype(np.float32)
    valid = flat > 0
    counts = valid.sum(axis=1)
    sums = np.where(valid, flat, 0.0).sum(axis=1)
    fallback = flat.mean(axis=1)
    spectrum = np.divide(sums, counts, out=fallback, where=counts > 0)
    spectrum[~np.isfinite(spectrum)] = 0.0
    return spectrum.astype(np.float32)


def make_radial_bins(height: int, width: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ids = np.floor(radius / (radius.max() + 1e-6) * bins).astype(np.int64)
    ids = np.clip(ids, 0, bins - 1)
    counts = np.bincount(ids.reshape(-1), minlength=bins).astype(np.float32)
    counts[counts == 0] = 1.0
    return ids, counts


def radial_profile_for_band(image: np.ndarray, bin_ids: np.ndarray, bin_counts: np.ndarray) -> np.ndarray:
    weights = image.reshape(-1).astype(np.float32)
    profile = np.bincount(bin_ids.reshape(-1), weights=weights, minlength=len(bin_counts))[: len(bin_counts)]
    profile = profile / bin_counts
    max_value = float(np.max(np.abs(profile)))
    if max_value > 0:
        profile = profile / max_value
    return profile.astype(np.float32)


def select_wavelengths_from_real_train(
    records: list[SampleRecord],
    template: EnviMeta,
    top_k: int,
    min_gap_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectra: list[np.ndarray] = []
    y: list[int] = []
    for idx, rec in enumerate(records, start=1):
        cube = read_fitted_cube(rec.hdr_path, template)
        spectra.append(mean_nonzero_spectrum(cube))
        y.append(rec.class_index)
        if idx % 50 == 0 or idx == len(records):
            print(f"[feature-select] spectra {idx}/{len(records)}")
    x = np.vstack(spectra)
    y_arr = np.asarray(y, dtype=int)
    f_values, _p_values = f_classif(x, y_arr)
    f_values = np.nan_to_num(f_values, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(f_values)[::-1]
    selected: list[int] = []
    for idx in order:
        wavelength = float(template.wavelengths[idx])
        if all(abs(wavelength - float(template.wavelengths[j])) >= min_gap_nm for j in selected):
            selected.append(int(idx))
        if len(selected) >= top_k:
            break
    selected_idx = np.asarray(selected, dtype=int)
    print("[feature-select] selected wavelengths:", ", ".join(f"{template.wavelengths[i]:.0f}" for i in selected_idx))
    return x, y_arr, selected_idx


def feature_from_cube(
    cube: np.ndarray,
    selected_idx: np.ndarray,
    radial_bins: int,
    bin_ids: np.ndarray,
    bin_counts: np.ndarray,
) -> np.ndarray:
    spectrum = mean_nonzero_spectrum(cube)
    selected_spectrum = spectrum[selected_idx]
    spectral_norm = (selected_spectrum - selected_spectrum.mean()) / (selected_spectrum.std() + 1e-6)
    radial_features = []
    for band_idx in selected_idx:
        radial_features.append(radial_profile_for_band(cube[int(band_idx)], bin_ids, bin_counts))
    radial = np.concatenate(radial_features) if radial_features else np.empty(0, dtype=np.float32)
    global_features = np.asarray(
        [
            float(spectrum.mean()),
            float(spectrum.std()),
            float(np.percentile(spectrum, 10)),
            float(np.percentile(spectrum, 90)),
            float(cube.mean()),
            float(cube.std()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([spectral_norm.astype(np.float32), radial.astype(np.float32), global_features])


def extract_features(
    records: list[SampleRecord],
    template: EnviMeta,
    selected_idx: np.ndarray,
    radial_bins: int,
    label: str,
) -> FeatureBundle:
    bin_ids, bin_counts = make_radial_bins(template.lines, template.samples, radial_bins)
    names: list[str] = []
    labels: list[str] = []
    y: list[int] = []
    features: list[np.ndarray] = []
    for idx, rec in enumerate(records, start=1):
        cube = read_fitted_cube(rec.hdr_path, template)
        features.append(feature_from_cube(cube, selected_idx, radial_bins, bin_ids, bin_counts))
        names.append(rec.name)
        labels.append(rec.label)
        y.append(rec.class_index)
        if idx % 50 == 0 or idx == len(records):
            print(f"[features] {label}: {idx}/{len(records)}")
    x = np.vstack(features).astype(np.float32) if features else np.empty((0, len(selected_idx) * (radial_bins + 1) + 6), dtype=np.float32)
    return FeatureBundle(names=names, labels=labels, y=np.asarray(y, dtype=int), x=x)


def spectral_shift(cube: np.ndarray, shift: float) -> np.ndarray:
    bands = cube.shape[0]
    positions = np.arange(bands, dtype=np.float32)
    source = np.clip(positions - shift, 0, bands - 1)
    left = np.floor(source).astype(np.int64)
    right = np.clip(left + 1, 0, bands - 1)
    weight = (source - left).astype(np.float32)[:, None, None]
    return cube[left] * (1.0 - weight) + cube[right] * weight


def augment_cube_classical(cube: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = cube.astype(np.float32, copy=True)
    k = int(rng.integers(0, 4))
    out = np.rot90(out, k, axes=(1, 2)).copy()
    if rng.random() < 0.5:
        out = out[:, :, ::-1].copy()
    if rng.random() < 0.5:
        out = out[:, ::-1, :].copy()
    out = spectral_shift(out, float(rng.uniform(-2.0, 2.0)))
    out *= float(rng.normal(1.0, 0.06))
    out += float(rng.normal(0.0, 0.008 * max(float(out.std()), 1.0)))
    noise_scale = 0.012 * max(float(out.std()), 1.0)
    out += rng.normal(0.0, noise_scale, size=out.shape).astype(np.float32)
    out[out < 0] = 0.0
    return out.astype(np.float32)


def build_classical_augmentation_features(
    train_records: list[SampleRecord],
    template: EnviMeta,
    selected_idx: np.ndarray,
    radial_bins: int,
    per_class: int,
    seed: int,
) -> FeatureBundle:
    if per_class <= 0:
        return FeatureBundle(names=[], labels=[], y=np.empty(0, dtype=int), x=np.empty((0, len(selected_idx) * (radial_bins + 1) + 6), dtype=np.float32))
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[SampleRecord]] = {idx: [] for idx in range(len(CATEGORY_SPECS))}
    for rec in train_records:
        by_class[rec.class_index].append(rec)
    bin_ids, bin_counts = make_radial_bins(template.lines, template.samples, radial_bins)
    names: list[str] = []
    labels: list[str] = []
    y: list[int] = []
    features: list[np.ndarray] = []
    for class_index, (_prefix, _folder, label) in enumerate(CATEGORY_SPECS):
        pool = by_class[class_index]
        if not pool:
            continue
        for item in range(1, per_class + 1):
            rec = pool[int(rng.integers(0, len(pool)))]
            cube = read_fitted_cube(rec.hdr_path, template)
            aug_cube = augment_cube_classical(cube, rng)
            features.append(feature_from_cube(aug_cube, selected_idx, radial_bins, bin_ids, bin_counts))
            names.append(f"classical_{label}_{item:04d}")
            labels.append(label)
            y.append(class_index)
        print(f"[classical-augment] {label}: {per_class}/{per_class}")
    return FeatureBundle(names=names, labels=labels, y=np.asarray(y, dtype=int), x=np.vstack(features).astype(np.float32))


def make_models(seed: int) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=seed),
        "svm_rbf": SVC(C=10.0, gamma="scale", class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def evaluate_scenario(
    method: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_names: list[str],
    test_labels: list[str],
    label_names: list[str],
    seed: int,
    out_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    reports: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    confusion_by_model: dict[str, np.ndarray] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for model_name, estimator in make_models(seed).items():
        clf = make_pipeline(StandardScaler(), estimator)
        clf.fit(train_x, train_y)
        pred = clf.predict(test_x)
        for sample_idx, (sample, true_idx, true_label, pred_idx) in enumerate(zip(test_names, test_y, test_labels, pred)):
            prediction_rows.append(
                {
                    "augmentation": method,
                    "classifier": model_name,
                    "sample_index": sample_idx,
                    "sample": sample,
                    "true_index": int(true_idx),
                    "true_label": true_label,
                    "predicted_index": int(pred_idx),
                    "predicted_label": label_names[int(pred_idx)],
                    "correct": int(int(true_idx) == int(pred_idx)),
                }
            )
        cm = confusion_matrix(test_y, pred, labels=np.arange(len(label_names)))
        confusion_by_model[model_name] = cm
        report = classification_report(
            test_y,
            pred,
            labels=np.arange(len(label_names)),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        rows.append(
            {
                "augmentation": method,
                "classifier": model_name,
                "train_n": int(train_x.shape[0]),
                "test_n": int(test_x.shape[0]),
                "accuracy": accuracy_score(test_y, pred),
                "balanced_accuracy": balanced_accuracy_score(test_y, pred),
                "macro_f1": f1_score(test_y, pred, average="macro"),
                "weighted_f1": f1_score(test_y, pred, average="weighted"),
                "mcc": matthews_corrcoef(test_y, pred),
            }
        )
        for class_name in label_names:
            class_report = report[class_name]
            reports.append(
                {
                    "augmentation": method,
                    "classifier": model_name,
                    "class": class_name,
                    "precision": class_report["precision"],
                    "recall": class_report["recall"],
                    "f1": class_report["f1-score"],
                    "support": class_report["support"],
                }
            )
        pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
            out_dir / f"confusion_{method}_{model_name}.csv",
            encoding="utf-8-sig",
        )
        print(f"[eval] {method}/{model_name}: accuracy={rows[-1]['accuracy']:.4f}, macro_f1={rows[-1]['macro_f1']:.4f}")
    return rows, confusion_by_model, reports, prediction_rows


def exact_mcnemar_pvalue(a_better: int, b_better: int) -> float:
    discordant = a_better + b_better
    if discordant == 0:
        return 1.0
    smaller = min(a_better, b_better)
    cdf = 0.0
    for k in range(smaller + 1):
        cdf += math.comb(discordant, k) * (0.5 ** discordant)
    return min(1.0, 2.0 * cdf)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, p_value) in enumerate(indexed):
        value = min(1.0, (m - rank) * p_value)
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def bootstrap_metric_differences(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    class_count: int,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    labels = np.arange(class_count)
    acc_a = accuracy_score(y_true, pred_a)
    acc_b = accuracy_score(y_true, pred_b)
    f1_a = f1_score(y_true, pred_a, average="macro", labels=labels, zero_division=0)
    f1_b = f1_score(y_true, pred_b, average="macro", labels=labels, zero_division=0)
    acc_deltas = np.empty(iterations, dtype=np.float64)
    f1_deltas = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        yy = y_true[idx]
        aa = pred_a[idx]
        bb = pred_b[idx]
        acc_deltas[i] = accuracy_score(yy, aa) - accuracy_score(yy, bb)
        f1_deltas[i] = f1_score(yy, aa, average="macro", labels=labels, zero_division=0) - f1_score(
            yy, bb, average="macro", labels=labels, zero_division=0
        )
    return {
        "accuracy_a": float(acc_a),
        "accuracy_b": float(acc_b),
        "accuracy_delta": float(acc_a - acc_b),
        "accuracy_delta_ci_low": float(np.percentile(acc_deltas, 2.5)),
        "accuracy_delta_ci_high": float(np.percentile(acc_deltas, 97.5)),
        "macro_f1_a": float(f1_a),
        "macro_f1_b": float(f1_b),
        "macro_f1_delta": float(f1_a - f1_b),
        "macro_f1_delta_ci_low": float(np.percentile(f1_deltas, 2.5)),
        "macro_f1_delta_ci_high": float(np.percentile(f1_deltas, 97.5)),
    }


def compute_significance_tests(
    predictions: pd.DataFrame,
    label_names: list[str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    comparison_pairs = [
        ("PINN-DDPM", "Real only"),
        ("PINN-DDPM", "Classical augmentation"),
        ("PINN-DDPM", "DDPM"),
        ("DDPM", "Real only"),
        ("Classical augmentation", "Real only"),
    ]
    classifiers = sorted(predictions["classifier"].unique())
    rows: list[dict[str, object]] = []
    for classifier in classifiers:
        frame = predictions[predictions["classifier"] == classifier]
        available = set(frame["augmentation"])
        for method_a, method_b in comparison_pairs:
            if method_a not in available or method_b not in available:
                continue
            a = frame[frame["augmentation"] == method_a].sort_values("sample_index")
            b = frame[frame["augmentation"] == method_b].sort_values("sample_index")
            if list(a["sample"]) != list(b["sample"]):
                raise ValueError(f"Sample order mismatch for {classifier}: {method_a} vs {method_b}")
            y_true = a["true_index"].to_numpy(dtype=int)
            pred_a = a["predicted_index"].to_numpy(dtype=int)
            pred_b = b["predicted_index"].to_numpy(dtype=int)
            correct_a = pred_a == y_true
            correct_b = pred_b == y_true
            a_better = int(np.sum(correct_a & ~correct_b))
            b_better = int(np.sum(~correct_a & correct_b))
            both_correct = int(np.sum(correct_a & correct_b))
            both_wrong = int(np.sum(~correct_a & ~correct_b))
            stats = bootstrap_metric_differences(
                y_true,
                pred_a,
                pred_b,
                class_count=len(label_names),
                iterations=iterations,
                seed=seed + len(rows) * 17,
            )
            rows.append(
                {
                    "classifier": classifier,
                    "method_a": method_a,
                    "method_b": method_b,
                    "comparison": f"{method_a} - {method_b}",
                    "test_n": int(len(y_true)),
                    "both_correct": both_correct,
                    "both_wrong": both_wrong,
                    "a_correct_b_wrong": a_better,
                    "a_wrong_b_correct": b_better,
                    "mcnemar_discordant_n": a_better + b_better,
                    "mcnemar_exact_p": exact_mcnemar_pvalue(a_better, b_better),
                    **stats,
                }
            )
    p_adjusted = holm_adjust([float(row["mcnemar_exact_p"]) for row in rows])
    for row, p_value in zip(rows, p_adjusted):
        row["mcnemar_holm_p"] = p_value
        row["significant_holm_0.05"] = p_value < 0.05
    return pd.DataFrame(rows)


def setup_nature_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
        }
    )


def save_pub_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def plot_confusion(ax: plt.Axes, cm: np.ndarray, label_names: list[str], title: str) -> None:
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, pad=4)
    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = norm[i, j]
            color = "white" if value > 0.55 else "#1f2933"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color=color, fontsize=6)
    return im


def make_nature_figure(
    results: pd.DataFrame,
    per_class: pd.DataFrame,
    confusion: dict[tuple[str, str], np.ndarray],
    label_names: list[str],
    primary_classifier: str,
    out_root: Path,
) -> Path:
    setup_nature_style()
    method_order = ["Real only", "Classical augmentation", "DDPM", "PINN-DDPM"]
    available = [m for m in method_order if m in set(results["augmentation"])]
    primary = results[results["classifier"] == primary_classifier].copy()
    primary["augmentation"] = pd.Categorical(primary["augmentation"], categories=method_order, ordered=True)
    primary = primary.sort_values("augmentation")
    per_class_primary = per_class[per_class["classifier"] == primary_classifier].copy()
    per_class_primary["augmentation"] = pd.Categorical(per_class_primary["augmentation"], categories=method_order, ordered=True)
    per_class_primary = per_class_primary.sort_values(["augmentation", "class"])

    palette = {
        "Real only": "#4B5563",
        "Classical augmentation": "#88A9C3",
        "DDPM": "#D6A06A",
        "PINN-DDPM": "#4C9A7B",
    }
    fig = plt.figure(figsize=(7.2, 5.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.35, hspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    x = np.arange(len(primary))
    colors = [palette.get(str(m), "#999999") for m in primary["augmentation"]]
    width = 0.34
    ax_a.bar(x - width / 2, primary["accuracy"], width=width, color=colors, alpha=0.75, label="Accuracy")
    ax_a.bar(x + width / 2, primary["macro_f1"], width=width, color=colors, alpha=1.0, hatch="///", label="Macro-F1")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(primary["augmentation"], rotation=28, ha="right")
    ax_a.set_ylim(0, 1.05)
    ax_a.set_ylabel("Score on held-out real test set")
    ax_a.set_title(f"Primary classifier: {primary_classifier}")
    ax_a.legend(loc="lower right", fontsize=6)
    panel_label(ax_a, "a")

    best_method = "PINN-DDPM" if "PINN-DDPM" in available else str(primary.sort_values("macro_f1", ascending=False).iloc[0]["augmentation"])
    cm = confusion.get((best_method, primary_classifier))
    if cm is not None:
        im = plot_confusion(ax_b, cm, label_names, f"{best_method} confusion matrix")
        cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
        cbar.set_label("Row-normalized rate")
    else:
        ax_b.axis("off")
        ax_b.text(0.5, 0.5, "Confusion matrix unavailable", ha="center", va="center")
    panel_label(ax_b, "b")

    pivot = per_class_primary.pivot_table(index="class", columns="augmentation", values="f1", observed=False)
    class_x = np.arange(len(label_names))
    bar_width = 0.8 / max(len(available), 1)
    for offset, method in enumerate(available):
        values = pivot[method].reindex(label_names).to_numpy(dtype=float)
        ax_c.bar(
            class_x - 0.4 + bar_width / 2 + offset * bar_width,
            values,
            width=bar_width,
            color=palette.get(method, "#999999"),
            label=method,
        )
    ax_c.set_xticks(class_x)
    ax_c.set_xticklabels(label_names)
    ax_c.set_ylim(0, 1.05)
    ax_c.set_ylabel("Per-class F1")
    ax_c.set_title("Class-wise performance")
    ax_c.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), handlelength=1.2)
    panel_label(ax_c, "c")

    train_counts = primary[["augmentation", "train_n"]].copy()
    real_n = int(results["test_n"].iloc[0]) if len(results) else 0
    ax_d.bar(
        np.arange(len(train_counts)),
        train_counts["train_n"],
        color=[palette.get(str(m), "#999999") for m in train_counts["augmentation"]],
    )
    ax_d.axhline(real_n, color="#555555", linewidth=0.8, linestyle=":")
    if len(train_counts) > 0:
        ax_d.text(
            len(train_counts) - 0.48,
            real_n + max(float(train_counts["train_n"].max()) * 0.025, 8.0),
            f"Held-out test n = {real_n}",
            ha="right",
            va="bottom",
            fontsize=6,
            color="#444444",
        )
    ax_d.set_xticks(np.arange(len(train_counts)))
    ax_d.set_xticklabels(train_counts["augmentation"], rotation=28, ha="right")
    ax_d.set_ylabel("Training samples")
    ax_d.set_title("Training-set composition")
    panel_label(ax_d, "d")

    out_stem = out_root / "figures" / "nature_style_augmentation_comparison"
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    save_pub_figure(fig, out_stem)
    plt.close(fig)
    return out_stem.with_suffix(".png")


def make_pca_source_plot(
    bundles: dict[str, FeatureBundle],
    out_root: Path,
    seed: int,
) -> None:
    setup_nature_style()
    frames = []
    for method, bundle in bundles.items():
        if bundle.x.size == 0:
            continue
        frame = pd.DataFrame(bundle.x)
        frame["source"] = method
        frame["label"] = bundle.labels
        frames.append(frame)
    if not frames:
        return
    data = pd.concat(frames, ignore_index=True)
    x = data.drop(columns=["source", "label"]).to_numpy(dtype=np.float32)
    x_scaled = StandardScaler().fit_transform(x)
    coords = PCA(n_components=2, random_state=seed).fit_transform(x_scaled)
    data["PC1"] = coords[:, 0]
    data["PC2"] = coords[:, 1]
    colors = {"real_train": "#4B5563", "real_test": "#111827", "classical": "#88A9C3", "DDPM": "#D6A06A", "PINN-DDPM": "#4C9A7B"}
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for source, group in data.groupby("source"):
        ax.scatter(group["PC1"], group["PC2"], s=10, alpha=0.62, label=source, color=colors.get(source, "#999999"), edgecolors="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Feature-space distribution")
    ax.legend(fontsize=6, markerscale=1.5)
    out_stem = out_root / "figures" / "feature_space_pca_sources"
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    save_pub_figure(fig, out_stem)
    plt.close(fig)


def write_feature_selection_csv(path: Path, template: EnviMeta, selected_idx: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "band_index", "wavelength_nm"])
        for rank, idx in enumerate(selected_idx, start=1):
            writer.writerow([rank, int(idx), float(template.wavelengths[idx])])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare real-only and augmented training strategies on held-out real ENVI test cubes.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pinn-dir", type=Path, default=DEFAULT_PINN_DIR)
    parser.add_argument("--ddpm-dir", type=Path, default=DEFAULT_DDPM_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--augmented-per-class", type=int, default=100)
    parser.add_argument("--classical-per-class", type=int, default=100)
    parser.add_argument("--top-wavelengths", type=int, default=40)
    parser.add_argument("--min-wavelength-gap-nm", type=float, default=5.0)
    parser.add_argument("--radial-bins", type=int, default=12)
    parser.add_argument("--primary-classifier", choices=["logistic_regression", "svm_rbf", "random_forest", "gradient_boosting"], default="svm_rbf")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "csv").mkdir(parents=True, exist_ok=True)
    (args.out_root / "figures").mkdir(parents=True, exist_ok=True)

    train_records = discover_real_records(args.data_root, test=False)
    test_records = discover_real_records(args.data_root, test=True)
    print(f"[data] real train: {len(train_records)} {summarize_records(train_records)}")
    print(f"[data] real test:  {len(test_records)} {summarize_records(test_records)}")
    template = choose_template_meta(train_records)

    _train_spectra, _train_y_for_selection, selected_idx = select_wavelengths_from_real_train(
        train_records,
        template,
        top_k=args.top_wavelengths,
        min_gap_nm=args.min_wavelength_gap_nm,
    )
    write_feature_selection_csv(args.out_root / "csv" / "selected_wavelengths_real_train_only.csv", template, selected_idx)

    train_real = extract_features(train_records, template, selected_idx, args.radial_bins, "real_train")
    test_real = extract_features(test_records, template, selected_idx, args.radial_bins, "real_test")
    classical = build_classical_augmentation_features(
        train_records,
        template,
        selected_idx,
        args.radial_bins,
        per_class=args.classical_per_class,
        seed=args.seed,
    )
    ddpm_records = discover_augmented_records(args.ddpm_dir, "DDPM", args.augmented_per_class)
    pinn_records = discover_augmented_records(args.pinn_dir, "PINN-DDPM", args.augmented_per_class)
    ddpm = extract_features(ddpm_records, template, selected_idx, args.radial_bins, "DDPM") if ddpm_records else FeatureBundle([], [], np.empty(0, dtype=int), np.empty((0, train_real.x.shape[1]), dtype=np.float32))
    pinn = extract_features(pinn_records, template, selected_idx, args.radial_bins, "PINN-DDPM") if pinn_records else FeatureBundle([], [], np.empty(0, dtype=int), np.empty((0, train_real.x.shape[1]), dtype=np.float32))

    label_names = [label for _prefix, _folder, label in CATEGORY_SPECS]
    scenarios: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Real only": (train_real.x, train_real.y),
        "Classical augmentation": (np.vstack([train_real.x, classical.x]), np.concatenate([train_real.y, classical.y])),
    }
    if ddpm.x.size:
        scenarios["DDPM"] = (np.vstack([train_real.x, ddpm.x]), np.concatenate([train_real.y, ddpm.y]))
    if pinn.x.size:
        scenarios["PINN-DDPM"] = (np.vstack([train_real.x, pinn.x]), np.concatenate([train_real.y, pinn.y]))

    result_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    confusion_lookup: dict[tuple[str, str], np.ndarray] = {}
    for method, (x_train, y_train) in scenarios.items():
        rows, cm_by_model, reports, predictions = evaluate_scenario(
            method,
            x_train,
            y_train,
            test_real.x,
            test_real.y,
            test_real.names,
            test_real.labels,
            label_names,
            args.seed,
            args.out_root / "csv",
        )
        result_rows.extend(rows)
        per_class_rows.extend(reports)
        prediction_rows.extend(predictions)
        for model_name, cm in cm_by_model.items():
            confusion_lookup[(method, model_name)] = cm

    results = pd.DataFrame(result_rows).sort_values(["classifier", "augmentation"])
    per_class = pd.DataFrame(per_class_rows).sort_values(["classifier", "augmentation", "class"])
    predictions = pd.DataFrame(prediction_rows).sort_values(["classifier", "augmentation", "sample_index"])
    significance = compute_significance_tests(
        predictions,
        label_names=label_names,
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    ).sort_values(["classifier", "method_a", "method_b"])
    results.to_csv(args.out_root / "csv" / "augmentation_method_comparison.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(args.out_root / "csv" / "augmentation_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(args.out_root / "csv" / "test_predictions.csv", index=False, encoding="utf-8-sig")
    significance.to_csv(args.out_root / "csv" / "augmentation_significance_tests.csv", index=False, encoding="utf-8-sig")
    summary = results.sort_values(["macro_f1", "accuracy"], ascending=False)
    summary.to_csv(args.out_root / "csv" / "augmentation_method_comparison_ranked.csv", index=False, encoding="utf-8-sig")

    figure_path = make_nature_figure(results, per_class, confusion_lookup, label_names, args.primary_classifier, args.out_root)
    make_pca_source_plot(
        {
            "real_train": train_real,
            "real_test": test_real,
            "classical": classical,
            "DDPM": ddpm,
            "PINN-DDPM": pinn,
        },
        args.out_root,
        args.seed,
    )
    print("\n[summary]")
    print(summary.to_string(index=False))
    print("\n[significance]")
    keep_cols = [
        "classifier",
        "comparison",
        "accuracy_delta",
        "accuracy_delta_ci_low",
        "accuracy_delta_ci_high",
        "macro_f1_delta",
        "macro_f1_delta_ci_low",
        "macro_f1_delta_ci_high",
        "mcnemar_exact_p",
        "mcnemar_holm_p",
    ]
    print(significance[keep_cols].to_string(index=False))
    print(f"\n[outputs] results: {args.out_root / 'csv' / 'augmentation_method_comparison.csv'}")
    print(f"[outputs] per-class: {args.out_root / 'csv' / 'augmentation_per_class_metrics.csv'}")
    print(f"[outputs] predictions: {args.out_root / 'csv' / 'test_predictions.csv'}")
    print(f"[outputs] significance: {args.out_root / 'csv' / 'augmentation_significance_tests.csv'}")
    print(f"[outputs] figure: {figure_path}")
    if "DDPM" not in scenarios:
        print(
            "\n[note] DDPM was not evaluated because the DDPM directory was not found or empty. "
            "Generate a no-physics DDPM set into ENVI_DDPM_AUGMENTED, then rerun this script."
        )


if __name__ == "__main__":
    main()
