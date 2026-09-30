from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.stats import binomtest
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent
CLASS_LABELS = ["CMB", "DQB", "DWB", "WQ", "YMXB"]
METHODS = ["ANOVA", "MI", "RF", "PLS-VIP", "SPA", "ReliefF", "2D-COS"]


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low) if high > low else np.zeros_like(values)


def score_anova(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scores, _ = f_classif(x, y)
    return minmax(scores)


def score_mi(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    return minmax(mutual_info_classif(x, y, random_state=seed))


def score_rf(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x, y)
    return minmax(model.feature_importances_)


def score_pls_vip(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    one_hot = np.eye(len(CLASS_LABELS), dtype=np.float64)[y]
    components = min(15, x.shape[0] - 1, x.shape[1])
    model = PLSRegression(n_components=components, scale=True)
    model.fit(x, one_hot)
    scores = model.x_scores_
    weights = model.x_weights_
    q = model.y_loadings_
    explained_y = np.sum(scores**2, axis=0) * np.sum(q**2, axis=0)
    weight_norm = np.sum(weights**2, axis=0)
    weight_norm[weight_norm == 0] = 1.0
    vip = np.sqrt(
        x.shape[1]
        * np.sum((weights**2 / weight_norm) * explained_y[None, :], axis=1)
        / max(float(explained_y.sum()), 1e-12)
    )
    return minmax(vip)


def score_spa(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    _q, _r, pivots = qr(centered, mode="economic", pivoting=True)
    scores = np.zeros(x.shape[1], dtype=np.float64)
    scores[pivots] = np.linspace(1.0, 0.0, len(pivots), endpoint=False)
    return scores


def score_relieff(x: np.ndarray, y: np.ndarray, seed: int, neighbors: int = 10) -> np.ndarray:
    scaled = StandardScaler().fit_transform(x)
    classes = np.unique(y)
    priors = {label: float(np.mean(y == label)) for label in classes}
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(y), size=min(200, len(y)), replace=False)
    nearest = NearestNeighbors(n_neighbors=len(y), metric="euclidean").fit(scaled)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    ranges = np.ptp(x, axis=0)
    ranges[ranges == 0] = 1.0
    for sample_index in sample_indices:
        indices = nearest.kneighbors(
            scaled[sample_index : sample_index + 1], return_distance=False
        )[0]
        hits = [index for index in indices if y[index] == y[sample_index] and index != sample_index][
            :neighbors
        ]
        if hits:
            weights -= np.mean(
                np.abs(x[sample_index] - x[hits]) / ranges,
                axis=0,
            ) / len(sample_indices)
        for label in classes:
            if label == y[sample_index]:
                continue
            misses = [index for index in indices if y[index] == label][:neighbors]
            if not misses:
                continue
            class_weight = priors[label] / max(1.0 - priors[y[sample_index]], 1e-12)
            weights += class_weight * np.mean(
                np.abs(x[sample_index] - x[misses]) / ranges,
                axis=0,
            ) / len(sample_indices)
    return minmax(np.maximum(weights, 0.0))


def score_2dcos(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    class_means = np.vstack([x[y == label].mean(axis=0) for label in np.unique(y)])
    dynamic = class_means - class_means.mean(axis=0, keepdims=True)
    synchronous = dynamic.T @ dynamic / max(dynamic.shape[0] - 1, 1)
    n_classes = dynamic.shape[0]
    noda = np.zeros((n_classes, n_classes), dtype=np.float64)
    for row in range(n_classes):
        for column in range(n_classes):
            if row != column:
                noda[row, column] = 1.0 / (np.pi * (column - row))
    asynchronous = dynamic.T @ noda @ dynamic / max(n_classes - 1, 1)
    diagonal = np.abs(np.diag(synchronous))
    synchronous_connectivity = np.sum(np.abs(synchronous), axis=1) - diagonal
    asynchronous_connectivity = np.sum(np.abs(asynchronous), axis=1)
    return (
        0.5 * minmax(diagonal)
        + 0.3 * minmax(synchronous_connectivity)
        + 0.2 * minmax(asynchronous_connectivity)
    )


def select_nonredundant(
    scores: np.ndarray,
    wavelengths: np.ndarray,
    count: int,
    min_gap_nm: float,
) -> np.ndarray:
    selected: list[int] = []
    for candidate in np.argsort(scores)[::-1]:
        if all(
            abs(float(wavelengths[candidate]) - float(wavelengths[index])) >= min_gap_nm
            for index in selected
        ):
            selected.append(int(candidate))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"Could select only {len(selected)} of {count} bands")
    return np.asarray(selected, dtype=np.int64)


def exact_mcnemar(
    y_true: np.ndarray, reference: np.ndarray, comparator: np.ndarray
) -> tuple[int, int, float]:
    reference_only = int(np.sum((reference == y_true) & (comparator != y_true)))
    comparator_only = int(np.sum((reference != y_true) & (comparator == y_true)))
    discordant = reference_only + comparator_only
    p_value = (
        float(binomtest(min(reference_only, comparator_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return reference_only, comparator_only, p_value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-free feature-wavelength comparison on the untouched ROI test set."
    )
    parser.add_argument(
        "--validation-root", type=Path, default=ROOT / "predeployment_validation"
    )
    parser.add_argument("--band-counts", nargs="*", type=int, default=[5, 10, 20, 30, 40])
    parser.add_argument("--min-gap-nm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    cache = args.validation_root / "cache"
    output = args.validation_root / "feature_wavelength_comparison"
    csv_dir = output / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    with np.load(cache / "real_train_extracted.npz", allow_pickle=False) as data:
        x_real = data["spectrum"].astype(np.float32)
        y_real = data["y"].astype(np.int64)
    with np.load(cache / "pinn_extracted.npz", allow_pickle=False) as data:
        x_synthetic = data["spectrum"].astype(np.float32)
        y_synthetic = data["y"].astype(np.int64)
    with np.load(cache / "real_test_extracted.npz", allow_pickle=False) as data:
        x_test = data["spectrum"].astype(np.float32)
        y_test = data["y"].astype(np.int64)
        test_names = data["names"].astype(str)
    run_config = json.loads(
        (args.validation_root / "run_config.json").read_text(encoding="utf-8")
    )
    wavelengths = np.arange(411.0, 780.0, 1.0, dtype=np.float64)
    if x_real.shape[1] != len(wavelengths):
        raise ValueError("Expected 369 spectral bands from 411 to 779 nm")

    score_map = {
        "ANOVA": score_anova(x_real, y_real),
        "MI": score_mi(x_real, y_real, args.seed),
        "RF": score_rf(x_real, y_real, args.seed),
        "PLS-VIP": score_pls_vip(x_real, y_real),
        "SPA": score_spa(x_real),
        "ReliefF": score_relieff(x_real, y_real, args.seed),
        "2D-COS": score_2dcos(x_real, y_real),
    }
    ranking_rows = []
    for method, scores in score_map.items():
        for rank, index in enumerate(np.argsort(scores)[::-1], start=1):
            ranking_rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "band_index": int(index),
                    "wavelength_nm": float(wavelengths[index]),
                    "score": float(scores[index]),
                }
            )
    pd.DataFrame(ranking_rows).to_csv(
        csv_dir / "all_method_rankings.csv", index=False, encoding="utf-8-sig"
    )

    x_train = np.vstack([x_real, x_synthetic])
    y_train = np.concatenate([y_real, y_synthetic])
    rows = []
    predictions: dict[str, np.ndarray] = {}
    selections = []
    for method in METHODS:
        for band_count in args.band_counts:
            selected = select_nonredundant(
                score_map[method], wavelengths, band_count, args.min_gap_nm
            )
            model = make_pipeline(
                StandardScaler(),
                SVC(C=10.0, gamma="scale", class_weight="balanced"),
            )
            model.fit(x_train[:, selected], y_train)
            prediction = model.predict(x_test[:, selected]).astype(np.int64)
            key = f"{method}_{band_count}"
            predictions[key] = prediction
            rows.append(
                {
                    "method": method,
                    "band_count": band_count,
                    "train_n": len(y_train),
                    "test_n": len(y_test),
                    "accuracy": accuracy_score(y_test, prediction),
                    "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                    "macro_f1": f1_score(y_test, prediction, average="macro"),
                    "mcc": matthews_corrcoef(y_test, prediction),
                    "selected_wavelengths_nm": ", ".join(
                        f"{wavelengths[index]:.0f}" for index in selected
                    ),
                }
            )
            for rank, index in enumerate(selected, start=1):
                selections.append(
                    {
                        "method": method,
                        "band_count": band_count,
                        "rank": rank,
                        "band_index": int(index),
                        "wavelength_nm": float(wavelengths[index]),
                        "score": float(score_map[method][index]),
                    }
                )
            print(
                f"[selection] {method} k={band_count}: "
                f"accuracy={rows[-1]['accuracy']:.4f}, macro_f1={rows[-1]['macro_f1']:.4f}"
            )

    full_model = make_pipeline(
        StandardScaler(),
        SVC(C=10.0, gamma="scale", class_weight="balanced"),
    )
    full_model.fit(x_train, y_train)
    full_prediction = full_model.predict(x_test).astype(np.int64)
    predictions["Full_spectrum_369"] = full_prediction
    rows.append(
        {
            "method": "Full spectrum",
            "band_count": 369,
            "train_n": len(y_train),
            "test_n": len(y_test),
            "accuracy": accuracy_score(y_test, full_prediction),
            "balanced_accuracy": balanced_accuracy_score(y_test, full_prediction),
            "macro_f1": f1_score(y_test, full_prediction, average="macro"),
            "mcc": matthews_corrcoef(y_test, full_prediction),
            "selected_wavelengths_nm": "411-779",
        }
    )

    results = pd.DataFrame(rows).sort_values(
        ["macro_f1", "accuracy"], ascending=False
    )
    results.to_csv(
        csv_dir / "feature_wavelength_independent_test_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(selections).to_csv(
        csv_dir / "selected_wavelengths_by_method_and_k.csv",
        index=False,
        encoding="utf-8-sig",
    )
    reference_row = results.iloc[0]
    reference_key = (
        "Full_spectrum_369"
        if reference_row["method"] == "Full spectrum"
        else f"{reference_row['method']}_{int(reference_row['band_count'])}"
    )
    significance = []
    for key, prediction in predictions.items():
        ref_only, comparator_only, p_value = exact_mcnemar(
            y_test, predictions[reference_key], prediction
        )
        significance.append(
            {
                "reference": reference_key,
                "comparator": key,
                "reference_only_correct": ref_only,
                "comparator_only_correct": comparator_only,
                "discordant_n": ref_only + comparator_only,
                "mcnemar_exact_p": p_value,
            }
        )
    pd.DataFrame(significance).to_csv(
        csv_dir / "feature_wavelength_mcnemar.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prediction_table = pd.DataFrame(
        {
            "sample": test_names,
            "true_index": y_test,
            "true_label": [CLASS_LABELS[index] for index in y_test],
        }
    )
    for key, prediction in predictions.items():
        prediction_table[key] = [CLASS_LABELS[index] for index in prediction]
    prediction_table.to_csv(
        csv_dir / "feature_wavelength_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "split": "70 real training and 30 untouched real test cubes per class",
        "selection_scope": "real training spectra only",
        "training_augmentation": "100 PINN-DDPM cubes per class added after selection",
        "classifier": "RBF-SVM with fixed hyperparameters",
        "best_method": str(reference_row["method"]),
        "best_band_count": int(reference_row["band_count"]),
        "best_accuracy": float(reference_row["accuracy"]),
        "best_macro_f1": float(reference_row["macro_f1"]),
        "unified_classification_config": run_config,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[best]")
    print(results.head(12).to_string(index=False))
    print(f"\n[outputs] {output}")


if __name__ == "__main__":
    main()
