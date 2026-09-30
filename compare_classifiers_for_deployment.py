from __future__ import annotations

import argparse
import os
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_augmentation_classification import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PINN_DIR,
    FeatureBundle,
    discover_augmented_records,
    discover_real_records,
    make_radial_bins,
    mean_nonzero_spectrum,
    radial_profile_for_band,
    read_fitted_cube,
    select_wavelengths_from_real_train,
    write_feature_selection_csv,
)
from pinn_diffusion_hsi_augmentation import CATEGORY_SPECS, EnviMeta, SampleRecord, choose_template_meta


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = ROOT / "deployment_model_selection_results"


@dataclass(frozen=True)
class FeatureSets:
    names: list[str]
    labels: list[str]
    y: np.ndarray
    x_by_variant: dict[str, np.ndarray]


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 96, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(32, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(32, hidden_dim // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ScaledTinyMLP(nn.Module):
    def __init__(self, model: TinyMLP, mean: np.ndarray, scale: np.ndarray):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(mean.astype(np.float32)))
        safe_scale = scale.astype(np.float32).copy()
        safe_scale[safe_scale == 0] = 1.0
        self.register_buffer("scale", torch.as_tensor(safe_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.scale)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_style() -> None:
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
        }
    )


def save_pub_figure(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=dpi, bbox_inches="tight")


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.10) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def feature_variants_from_cube(
    cube: np.ndarray,
    selected_idx: np.ndarray,
    radial_bins: int,
    bin_ids: np.ndarray,
    bin_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    spectrum = mean_nonzero_spectrum(cube)
    selected = spectrum[selected_idx].astype(np.float32)
    selected_norm = ((selected - selected.mean()) / (selected.std() + 1e-6)).astype(np.float32)
    radial = np.concatenate(
        [radial_profile_for_band(cube[int(band_idx)], bin_ids, bin_counts) for band_idx in selected_idx]
    ).astype(np.float32)
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
    return {
        "spectral_40": selected_norm,
        "spectral_radial": np.concatenate([selected_norm, radial]).astype(np.float32),
        "spectral_radial_stats": np.concatenate([selected_norm, radial, global_features]).astype(np.float32),
    }


def extract_feature_sets(
    records: list[SampleRecord],
    template: EnviMeta,
    selected_idx: np.ndarray,
    radial_bins: int,
    label: str,
) -> FeatureSets:
    bin_ids, bin_counts = make_radial_bins(template.lines, template.samples, radial_bins)
    names: list[str] = []
    labels: list[str] = []
    y: list[int] = []
    stores: dict[str, list[np.ndarray]] = {
        "spectral_40": [],
        "spectral_radial": [],
        "spectral_radial_stats": [],
    }
    for idx, rec in enumerate(records, start=1):
        cube = read_fitted_cube(rec.hdr_path, template)
        variants = feature_variants_from_cube(cube, selected_idx, radial_bins, bin_ids, bin_counts)
        for key, values in variants.items():
            stores[key].append(values)
        names.append(rec.name)
        labels.append(rec.label)
        y.append(rec.class_index)
        if idx % 50 == 0 or idx == len(records):
            print(f"[features] {label}: {idx}/{len(records)}")
    x_by_variant = {key: np.vstack(values).astype(np.float32) for key, values in stores.items()}
    return FeatureSets(names=names, labels=labels, y=np.asarray(y, dtype=int), x_by_variant=x_by_variant)


def sklearn_models(seed: int) -> dict[str, object]:
    return {
        "logistic_regression_l2": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=seed),
        ),
        "linear_svm": make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced", dual=False, max_iter=20000, random_state=seed),
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(),
            SVC(C=10.0, gamma="scale", class_weight="balanced", random_state=seed),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "knn_5": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
    }


def save_pickle_model(model: object, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path.stat().st_size


def benchmark_predict(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    x_test: np.ndarray,
    repeats: int,
    warmup: int = 20,
) -> tuple[float, float]:
    repeats = max(20, repeats)
    for idx in range(min(warmup, len(x_test))):
        _ = predict_fn(x_test[idx : idx + 1])
    t0 = time.perf_counter()
    for idx in range(repeats):
        sample = x_test[idx % len(x_test) : idx % len(x_test) + 1]
        _ = predict_fn(sample)
    single_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    batch_repeats = max(10, repeats // 20)
    _ = predict_fn(x_test)
    t0 = time.perf_counter()
    for _idx in range(batch_repeats):
        _ = predict_fn(x_test)
    batch_ms_per_sample = (time.perf_counter() - t0) * 1000.0 / (batch_repeats * len(x_test))
    return single_ms, batch_ms_per_sample


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for label in label_names:
        rows.append(
            {
                "class": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": report[label]["support"],
            }
        )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    return metrics, pd.DataFrame(rows), cm


def train_torch_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, TinyMLP, StandardScaler, dict[str, float]]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.18, random_state=seed)
    fit_idx, val_idx = next(splitter.split(train_x, train_y))
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(train_x[fit_idx]).astype(np.float32)
    x_val = scaler.transform(train_x[val_idx]).astype(np.float32)
    y_fit = train_y[fit_idx].astype(np.int64)
    y_val = train_y[val_idx].astype(np.int64)
    x_test_scaled = scaler.transform(test_x).astype(np.float32)

    model = TinyMLP(train_x.shape[1], len(CATEGORY_SPECS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_x_t = torch.from_numpy(x_val).to(device)
    val_y_t = torch.from_numpy(y_val).to(device)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_x_t), val_y_t).item())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test_scaled).to(device))
        pred = logits.argmax(dim=1).detach().cpu().numpy().astype(int)
    history = {"best_val_loss": best_val, "best_epoch": float(best_epoch), "epochs_ran": float(epoch)}
    return pred, model, scaler, history


def torch_predict_fn(model: TinyMLP, scaler: StandardScaler, device: torch.device) -> Callable[[np.ndarray], np.ndarray]:
    def predict(x: np.ndarray) -> np.ndarray:
        x_scaled = scaler.transform(x).astype(np.float32)
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x_scaled).to(device))
            return logits.argmax(dim=1).detach().cpu().numpy().astype(int)

    return predict


def save_torch_exports(
    model: TinyMLP,
    scaler: StandardScaler,
    input_dim: int,
    out_dir: Path,
    stem: str,
    device: torch.device,
) -> tuple[int, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / f"{stem}.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "num_classes": len(CATEGORY_SPECS),
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "class_labels": [label for _prefix, _folder, label in CATEGORY_SPECS],
    }
    torch.save(payload, state_path)
    export_notes = ["torch_state_dict"]

    model.eval()
    wrapped = ScaledTinyMLP(model, scaler.mean_, scaler.scale_).to(device)
    wrapped.eval()
    dummy = torch.zeros(1, input_dim, dtype=torch.float32).to(device)
    script_path = out_dir / f"{stem}_traced.pt"
    traced = torch.jit.trace(wrapped, dummy)
    traced.save(str(script_path))
    export_notes.append("torchscript")

    onnx_path = out_dir / f"{stem}.onnx"
    try:
        torch.onnx.export(
            wrapped,
            dummy,
            str(onnx_path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        export_notes.append("onnx")
    except Exception as exc:
        export_notes.append(f"onnx_failed:{type(exc).__name__}")
    size = sum(path.stat().st_size for path in [state_path, script_path, onnx_path] if path.exists())
    return size, ";".join(export_notes)


def deployment_metadata(model_name: str) -> dict[str, object]:
    meta = {
        "logistic_regression_l2": ("ONNX/manual linear model", True, "excellent"),
        "linear_svm": ("ONNX/manual linear decision function", True, "excellent"),
        "svm_rbf": ("sklearn/joblib CPU; ONNX possible but not TensorRT-friendly", False, "medium"),
        "random_forest": ("sklearn/joblib CPU or Treelite-style tree deployment", False, "medium"),
        "gradient_boosting": ("sklearn/joblib CPU or tree deployment", False, "medium"),
        "knn_5": ("stores training features; latency grows with training set", False, "low"),
        "tiny_mlp": ("TorchScript/ONNX/TensorRT candidate", True, "excellent"),
    }
    path, trt, rating = meta[model_name]
    return {"deployment_path": path, "tensorrt_friendly": trt, "deployment_rating": rating}


def recommendation_score(row: dict[str, object]) -> float:
    score = 100.0 * float(row["macro_f1"])
    score += 2.5 if bool(row["tensorrt_friendly"]) else 0.0
    score -= 0.35 * np.log1p(float(row["model_size_kb"]))
    score -= 0.08 * float(row["single_sample_latency_ms"])
    if row["model"] == "knn_5":
        score -= 3.0
    return float(score)


def make_summary_figure(results: pd.DataFrame, out_root: Path) -> None:
    setup_style()
    ranked = results.sort_values("macro_f1", ascending=False).head(12).copy()
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.46)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    variant_short = {
        "spectral_40": "Spec40",
        "spectral_radial": "Spec+radial",
        "spectral_radial_stats": "Spec+radial+stats",
    }
    model_short = {
        "logistic_regression_l2": "LogReg",
        "linear_svm": "Linear SVM",
        "svm_rbf": "RBF-SVM",
        "random_forest": "RF",
        "gradient_boosting": "GB",
        "knn_5": "KNN",
        "tiny_mlp": "Tiny MLP",
    }
    labels = ranked.apply(lambda row: f"{variant_short.get(row['input_variant'], row['input_variant'])} | {model_short.get(row['model'], row['model'])}", axis=1)
    colors = np.where(ranked["tensorrt_friendly"].to_numpy(dtype=bool), "#4C9A7B", "#D6A06A")
    y = np.arange(len(ranked))[::-1]
    ax_a.barh(y, ranked["macro_f1"], color=colors, height=0.72)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=5.8)
    ax_a.set_xlim(max(0.75, ranked["macro_f1"].min() - 0.03), 1.01)
    ax_a.set_xlabel("Macro-F1")
    ax_a.set_title("Top held-out test performance")
    panel_label(ax_a, "a")

    scatter_colors = {"spectral_40": "#88A9C3", "spectral_radial": "#D6A06A", "spectral_radial_stats": "#4C9A7B"}
    for variant, group in results.groupby("input_variant"):
        ax_b.scatter(
            group["single_sample_latency_ms"],
            group["macro_f1"],
            s=np.clip(np.sqrt(group["model_size_kb"]) * 6, 18, 120),
            color=scatter_colors.get(variant, "#999999"),
            alpha=0.76,
            label=variant,
            edgecolors="white",
            linewidths=0.4,
        )
    ax_b.set_xscale("log")
    ax_b.set_xlabel("Single-sample prediction latency (ms, CPU)")
    ax_b.set_ylabel("Macro-F1")
    ax_b.set_title("Accuracy-latency-size trade-off")
    ax_b.legend(fontsize=6, loc="lower right")
    panel_label(ax_b, "b")

    deploy = results.sort_values("deployment_score", ascending=False).head(10)
    labels_c = deploy.apply(lambda row: f"{variant_short.get(row['input_variant'], row['input_variant'])}\n{model_short.get(row['model'], row['model'])}", axis=1)
    x = np.arange(len(deploy))
    ax_c.plot(x, deploy["macro_f1"], color="#1F2933", marker="o", linewidth=1.2, label="Macro-F1")
    ax_c.set_ylim(0.75, 1.01)
    ax_c.set_ylabel("Macro-F1")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(labels_c, rotation=34, ha="right", fontsize=6)
    ax_c2 = ax_c.twinx()
    ax_c2.bar(x, deploy["deployment_score"], color="#C7D2FE", alpha=0.55, label="Deployment score")
    ax_c2.set_ylabel("Deployment score")
    ax_c.set_title("Deployment-oriented shortlist")
    panel_label(ax_c, "c")

    out = out_root / "figures" / "deployment_model_selection"
    save_pub_figure(fig, out)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare classifiers for held-out accuracy and Jetson-style deployment readiness.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--pinn-dir", type=Path, default=DEFAULT_PINN_DIR)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--augmented-per-class", type=int, default=100)
    parser.add_argument("--top-wavelengths", type=int, default=40)
    parser.add_argument("--min-wavelength-gap-nm", type=float, default=5.0)
    parser.add_argument("--radial-bins", type=int, default=12)
    parser.add_argument("--latency-repeats", type=int, default=500)
    parser.add_argument("--mlp-epochs", type=int, default=350)
    parser.add_argument("--mlp-patience", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    csv_dir = args.out_root / "csv"
    model_dir = args.out_root / "models"
    fig_dir = args.out_root / "figures"
    for directory in [csv_dir, model_dir, fig_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    label_names = [label for _prefix, _folder, label in CATEGORY_SPECS]
    train_records = discover_real_records(args.data_root, test=False)
    test_records = discover_real_records(args.data_root, test=True)
    pinn_records = discover_augmented_records(args.pinn_dir, "PINN-DDPM", args.augmented_per_class)
    template = choose_template_meta(train_records)
    _spectra, _y, selected_idx = select_wavelengths_from_real_train(
        train_records,
        template,
        top_k=args.top_wavelengths,
        min_gap_nm=args.min_wavelength_gap_nm,
    )
    write_feature_selection_csv(csv_dir / "selected_wavelengths_real_train_only.csv", template, selected_idx)

    real_train = extract_feature_sets(train_records, template, selected_idx, args.radial_bins, "real_train")
    pinn_train = extract_feature_sets(pinn_records, template, selected_idx, args.radial_bins, "PINN-DDPM")
    real_test = extract_feature_sets(test_records, template, selected_idx, args.radial_bins, "real_test")

    selected_frame = pd.DataFrame(
        {
            "rank": np.arange(1, len(selected_idx) + 1),
            "band_index": selected_idx,
            "wavelength_nm": template.wavelengths[selected_idx],
        }
    )
    selected_frame.to_csv(csv_dir / "deployment_selected_wavelengths.csv", index=False, encoding="utf-8-sig")

    rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] torch={device}")

    for variant in ["spectral_40", "spectral_radial", "spectral_radial_stats"]:
        train_x = np.vstack([real_train.x_by_variant[variant], pinn_train.x_by_variant[variant]]).astype(np.float32)
        train_y = np.concatenate([real_train.y, pinn_train.y]).astype(int)
        test_x = real_test.x_by_variant[variant].astype(np.float32)
        test_y = real_test.y.astype(int)
        print(f"[variant] {variant}: train={train_x.shape}, test={test_x.shape}")

        for model_name, model in sklearn_models(args.seed).items():
            model.fit(train_x, train_y)
            pred = model.predict(test_x).astype(int)
            metrics, report, cm = evaluate_predictions(test_y, pred, label_names)
            model_path = model_dir / f"{variant}_{model_name}.pkl"
            size_bytes = save_pickle_model(model, model_path)
            single_ms, batch_ms = benchmark_predict(model.predict, test_x, args.latency_repeats)
            row = {
                "input_variant": variant,
                "model": model_name,
                "train_n": int(train_x.shape[0]),
                "test_n": int(test_x.shape[0]),
                "feature_dim": int(train_x.shape[1]),
                "model_size_kb": size_bytes / 1024.0,
                "single_sample_latency_ms": single_ms,
                "batch_latency_ms_per_sample": batch_ms,
                "export_artifact": str(model_path),
                "notes": "sklearn_pickle",
                **deployment_metadata(model_name),
                **metrics,
            }
            row["deployment_score"] = recommendation_score(row)
            rows.append(row)
            report.insert(0, "input_variant", variant)
            report.insert(1, "model", model_name)
            per_class_rows.extend(report.to_dict(orient="records"))
            pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
                csv_dir / f"confusion_{variant}_{model_name}.csv", encoding="utf-8-sig"
            )
            for sample_idx, (sample, true_idx, true_label, pred_idx) in enumerate(
                zip(real_test.names, test_y, real_test.labels, pred)
            ):
                prediction_rows.append(
                    {
                        "input_variant": variant,
                        "model": model_name,
                        "sample_index": sample_idx,
                        "sample": sample,
                        "true_index": int(true_idx),
                        "true_label": true_label,
                        "predicted_index": int(pred_idx),
                        "predicted_label": label_names[int(pred_idx)],
                        "correct": int(int(true_idx) == int(pred_idx)),
                    }
                )
            print(f"[eval] {variant}/{model_name}: acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, latency={single_ms:.3f} ms")

        pred, mlp, scaler, history = train_torch_mlp(
            train_x,
            train_y,
            test_x,
            seed=args.seed,
            epochs=args.mlp_epochs,
            patience=args.mlp_patience,
            batch_size=args.batch_size,
            device=device,
        )
        metrics, report, cm = evaluate_predictions(test_y, pred, label_names)
        size_bytes, export_notes = save_torch_exports(
            mlp,
            scaler,
            input_dim=train_x.shape[1],
            out_dir=model_dir,
            stem=f"{variant}_tiny_mlp",
            device=device,
        )
        predict_fn = torch_predict_fn(mlp, scaler, device)
        single_ms, batch_ms = benchmark_predict(predict_fn, test_x, args.latency_repeats)
        row = {
            "input_variant": variant,
            "model": "tiny_mlp",
            "train_n": int(train_x.shape[0]),
            "test_n": int(test_x.shape[0]),
            "feature_dim": int(train_x.shape[1]),
            "model_size_kb": size_bytes / 1024.0,
            "single_sample_latency_ms": single_ms,
            "batch_latency_ms_per_sample": batch_ms,
            "export_artifact": str(model_dir / f"{variant}_tiny_mlp.pt"),
            "notes": export_notes + f";best_epoch={history['best_epoch']:.0f};best_val_loss={history['best_val_loss']:.4f}",
            **deployment_metadata("tiny_mlp"),
            **metrics,
        }
        row["deployment_score"] = recommendation_score(row)
        rows.append(row)
        report.insert(0, "input_variant", variant)
        report.insert(1, "model", "tiny_mlp")
        per_class_rows.extend(report.to_dict(orient="records"))
        pd.DataFrame(cm, index=label_names, columns=label_names).to_csv(
            csv_dir / f"confusion_{variant}_tiny_mlp.csv", encoding="utf-8-sig"
        )
        for sample_idx, (sample, true_idx, true_label, pred_idx) in enumerate(
            zip(real_test.names, test_y, real_test.labels, pred)
        ):
            prediction_rows.append(
                {
                    "input_variant": variant,
                    "model": "tiny_mlp",
                    "sample_index": sample_idx,
                    "sample": sample,
                    "true_index": int(true_idx),
                    "true_label": true_label,
                    "predicted_index": int(pred_idx),
                    "predicted_label": label_names[int(pred_idx)],
                    "correct": int(int(true_idx) == int(pred_idx)),
                }
            )
        print(f"[eval] {variant}/tiny_mlp: acc={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, latency={single_ms:.3f} ms")

    results = pd.DataFrame(rows).sort_values(["macro_f1", "deployment_score"], ascending=False)
    per_class = pd.DataFrame(per_class_rows).sort_values(["input_variant", "model", "class"])
    predictions = pd.DataFrame(prediction_rows).sort_values(["input_variant", "model", "sample_index"])
    results.to_csv(csv_dir / "deployment_model_comparison.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(csv_dir / "deployment_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(csv_dir / "deployment_test_predictions.csv", index=False, encoding="utf-8-sig")
    make_summary_figure(results, args.out_root)
    print("\n[ranked]")
    columns = [
        "input_variant",
        "model",
        "accuracy",
        "macro_f1",
        "mcc",
        "feature_dim",
        "model_size_kb",
        "single_sample_latency_ms",
        "batch_latency_ms_per_sample",
        "tensorrt_friendly",
        "deployment_score",
        "notes",
    ]
    print(results[columns].head(20).to_string(index=False))
    print(f"\n[outputs] {csv_dir / 'deployment_model_comparison.csv'}")
    print(f"[outputs] {args.out_root / 'figures' / 'deployment_model_selection.png'}")


if __name__ == "__main__":
    main()
