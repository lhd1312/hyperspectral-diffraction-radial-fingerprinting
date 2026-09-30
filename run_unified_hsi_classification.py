from __future__ import annotations

import argparse
import json
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluate_augmentation_classification import (
    discover_augmented_records,
    discover_real_records,
    mean_nonzero_spectrum,
    read_fitted_cube,
)
from pinn_diffusion_hsi_augmentation import choose_template_meta


ROOT = Path(__file__).resolve().parent
CLASS_LABELS = ["CMB", "DQB", "DWB", "WQ", "YMXB"]
SEEDS = [2026, 2027, 2028]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def radial_geometry(height: int, width: int, bins: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:height, :width]
    radius = np.sqrt((yy - (height - 1) / 2.0) ** 2 + (xx - (width - 1) / 2.0) ** 2)
    ids = np.floor(radius / (radius.max() + 1e-6) * bins).astype(np.int64)
    ids = np.clip(ids, 0, bins - 1)
    counts = np.bincount(ids.reshape(-1), minlength=bins).astype(np.float32)
    counts[counts == 0] = 1.0
    return ids, counts


def radial_profile(image: np.ndarray, ids: np.ndarray, counts: np.ndarray) -> np.ndarray:
    profile = np.bincount(
        ids.reshape(-1),
        weights=image.reshape(-1).astype(np.float32),
        minlength=len(counts),
    )[: len(counts)]
    profile = profile / counts
    scale = float(np.max(np.abs(profile)))
    if scale > 0:
        profile = profile / scale
    return profile.astype(np.float32)


def robust01(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    low, high = np.percentile(array, [1.0, 99.0])
    if high <= low:
        low = float(array.min())
        high = float(array.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def resize_nearest_stack(stack: np.ndarray, size: int) -> np.ndarray:
    y_idx = np.linspace(0, stack.shape[-2] - 1, size).round().astype(np.int64)
    x_idx = np.linspace(0, stack.shape[-1] - 1, size).round().astype(np.int64)
    return stack[..., y_idx[:, None], x_idx[None, :]]


def select_wavelengths(
    spectra: np.ndarray,
    labels: np.ndarray,
    wavelengths: np.ndarray,
    top_k: int,
    min_gap_nm: float,
) -> tuple[np.ndarray, np.ndarray]:
    scores, _ = f_classif(spectra, labels)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    selected: list[int] = []
    for candidate in np.argsort(scores)[::-1]:
        wavelength = float(wavelengths[candidate])
        if all(abs(wavelength - float(wavelengths[index])) >= min_gap_nm for index in selected):
            selected.append(int(candidate))
        if len(selected) == top_k:
            break
    if len(selected) != top_k:
        raise RuntimeError(f"Only {len(selected)} wavelengths satisfy min_gap_nm={min_gap_nm}")
    return np.asarray(selected, dtype=np.int64), scores


@dataclass
class Extracted:
    names: list[str]
    y: np.ndarray
    spectrum: np.ndarray
    single: np.ndarray
    pseudo_features: np.ndarray
    fusion: np.ndarray
    rgb: np.ndarray
    cube3d: np.ndarray


def extract_records(
    records,
    template,
    selected_idx: np.ndarray,
    best_band_idx: int,
    radial_bins: int,
    image_size: int,
    cube_size: int,
    cube_bands: int,
    label: str,
) -> Extracted:
    ids, counts = radial_geometry(template.lines, template.samples, radial_bins)
    cube_indices = np.linspace(0, template.bands - 1, cube_bands).round().astype(np.int64)
    context_indices = np.linspace(0, template.bands - 1, 24).round().astype(np.int64)
    names: list[str] = []
    labels: list[int] = []
    spectra: list[np.ndarray] = []
    single_features: list[np.ndarray] = []
    pseudo_features: list[np.ndarray] = []
    fusion_features: list[np.ndarray] = []
    rgb_images: list[np.ndarray] = []
    cube_images: list[np.ndarray] = []

    for number, record in enumerate(records, start=1):
        cube = read_fitted_cube(record.hdr_path, template)
        spectrum = mean_nonzero_spectrum(cube)
        selected_spectrum = spectrum[selected_idx]
        selected_snv = (selected_spectrum - selected_spectrum.mean()) / (
            selected_spectrum.std() + 1e-6
        )
        full_snv = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-6)

        selected_radial = np.concatenate(
            [radial_profile(cube[index], ids, counts) for index in selected_idx]
        )
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
        fusion = np.concatenate([selected_snv, selected_radial, global_features]).astype(
            np.float32
        )

        single_image = cube[best_band_idx]
        single_profile = radial_profile(single_image, ids, counts)
        single_stats = np.asarray(
            [
                single_image.mean(),
                single_image.std(),
                np.percentile(single_image, 10),
                np.percentile(single_image, 90),
                single_image[template.lines // 2, template.samples // 2],
            ],
            dtype=np.float32,
        )
        single_feature = np.concatenate([single_stats, single_profile]).astype(np.float32)

        channel_1 = robust01(single_image)
        mean_image = cube[context_indices].mean(axis=0)
        channel_2 = robust01(mean_image)
        grad_y, grad_x = np.gradient(channel_2)
        channel_3 = robust01(np.hypot(grad_x, grad_y))
        pseudo = np.stack([channel_1, channel_2, channel_3])
        pseudo_radial = np.concatenate(
            [radial_profile(channel, ids, counts) for channel in pseudo]
        )
        pseudo_stats = np.asarray(
            [
                value
                for channel in pseudo
                for value in (
                    channel.mean(),
                    channel.std(),
                    np.percentile(channel, 10),
                    np.percentile(channel, 90),
                )
            ],
            dtype=np.float32,
        )

        cube_small = resize_nearest_stack(cube[cube_indices], cube_size)
        cube_small = robust01(cube_small)

        names.append(record.name)
        labels.append(record.class_index)
        spectra.append(full_snv.astype(np.float32))
        single_features.append(single_feature)
        pseudo_features.append(np.concatenate([pseudo_stats, pseudo_radial]))
        fusion_features.append(fusion)
        rgb_images.append(resize_nearest_stack(pseudo, image_size).astype(np.float16))
        cube_images.append(cube_small.astype(np.float16))
        if number % 50 == 0 or number == len(records):
            print(f"[extract] {label}: {number}/{len(records)}")

    return Extracted(
        names=names,
        y=np.asarray(labels, dtype=np.int64),
        spectrum=np.vstack(spectra),
        single=np.vstack(single_features),
        pseudo_features=np.vstack(pseudo_features),
        fusion=np.vstack(fusion_features),
        rgb=np.stack(rgb_images),
        cube3d=np.stack(cube_images)[:, None],
    )


def save_extracted(path: Path, data: Extracted) -> None:
    np.savez_compressed(
        path,
        names=np.asarray(data.names),
        y=data.y,
        spectrum=data.spectrum,
        single=data.single,
        pseudo_features=data.pseudo_features,
        fusion=data.fusion,
        rgb=data.rgb,
        cube3d=data.cube3d,
    )


def load_extracted(path: Path) -> Extracted:
    with np.load(path, allow_pickle=False) as data:
        return Extracted(
            names=data["names"].astype(str).tolist(),
            y=data["y"].astype(np.int64),
            spectrum=data["spectrum"].astype(np.float32),
            single=data["single"].astype(np.float32),
            pseudo_features=data["pseudo_features"].astype(np.float32),
            fusion=data["fusion"].astype(np.float32),
            rgb=data["rgb"],
            cube3d=data["cube3d"],
        )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    accuracy: list[float] = []
    macro_f1: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(y_true), len(y_true))
        accuracy.append(accuracy_score(y_true[indices], y_pred[indices]))
        macro_f1.append(f1_score(y_true[indices], y_pred[indices], average="macro"))
    return {
        "accuracy_ci_low": float(np.percentile(accuracy, 2.5)),
        "accuracy_ci_high": float(np.percentile(accuracy, 97.5)),
        "macro_f1_ci_low": float(np.percentile(macro_f1, 2.5)),
        "macro_f1_ci_high": float(np.percentile(macro_f1, 97.5)),
    }


def exact_mcnemar(
    y_true: np.ndarray,
    reference_pred: np.ndarray,
    comparator_pred: np.ndarray,
) -> tuple[int, int, float]:
    reference_correct = reference_pred == y_true
    comparator_correct = comparator_pred == y_true
    ref_only = int(np.sum(reference_correct & ~comparator_correct))
    comp_only = int(np.sum(~reference_correct & comparator_correct))
    discordant = ref_only + comp_only
    p_value = (
        float(binomtest(min(ref_only, comp_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return ref_only, comp_only, p_value


class ArrayDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        architecture: str,
        training: bool,
        seed: int,
    ) -> None:
        self.x = x
        self.y = y
        self.architecture = architecture
        self.training = training
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        value = np.asarray(self.x[index], dtype=np.float32).copy()
        if self.training:
            if self.architecture == "1D-CNN":
                value *= float(self.rng.uniform(0.97, 1.03))
                value += self.rng.normal(0.0, 0.008, size=value.shape).astype(np.float32)
            else:
                if self.rng.random() < 0.5:
                    value = value[..., ::-1].copy()
                if self.rng.random() < 0.5:
                    value = value[..., ::-1, :].copy()
                rotations = int(self.rng.integers(0, 4))
                value = np.rot90(value, rotations, axes=(-2, -1)).copy()
        return torch.from_numpy(value), torch.tensor(int(self.y[index]), dtype=torch.long)


class CNN1D(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class CNN2D(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


class CNN3D(nn.Module):
    def __init__(self, classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.classifier = nn.Linear(32, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def build_neural_model(name: str, classes: int) -> nn.Module:
    if name == "1D-CNN":
        return CNN1D(classes)
    if name == "2D-CNN":
        return CNN2D(classes)
    if name == "3D-CNN":
        return CNN3D(classes)
    raise ValueError(name)


def predict_neural(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    architecture: str,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    loader = DataLoader(
        ArrayDataset(x, y, architecture, False, 0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    predictions: list[np.ndarray] = []
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for values, _labels in loader:
            values = values.to(device, non_blocking=True)
            predictions.append(model(values).argmax(dim=1).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return np.concatenate(predictions), elapsed * 1000.0 / len(y)


def train_neural(
    architecture: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, dict[str, float]]:
    seed_everything(seed)
    model = build_neural_model(architecture, len(CLASS_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    train_loader = DataLoader(
        ArrayDataset(x_train, y_train, architecture, True, seed),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    best_state = deepcopy(model.state_dict())
    best_val = -math.inf
    stale = 0
    best_epoch = 0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        for values, labels in train_loader:
            values = values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(values), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        val_pred, _ = predict_neural(
            model, x_val, y_val, architecture, device, batch_size
        )
        val_score = f1_score(y_val, val_pred, average="macro")
        if val_score > best_val + 1e-5:
            best_val = float(val_score)
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    test_pred, inference_ms = predict_neural(
        model, x_test, y_test, architecture, device, batch_size
    )
    metrics = classification_metrics(y_test, test_pred)
    metrics.update(
        {
            "best_epoch": float(best_epoch),
            "validation_macro_f1": best_val,
            "parameters": float(sum(parameter.numel() for parameter in model.parameters())),
            "inference_ms_per_cube": float(inference_ms),
        }
    )
    return model, test_pred, metrics


def majority_vote(predictions: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(predictions)
    output = np.empty(stacked.shape[1], dtype=np.int64)
    for index in range(stacked.shape[1]):
        output[index] = np.bincount(
            stacked[:, index], minlength=len(CLASS_LABELS)
        ).argmax()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified independent-test classification and hyperspectral-value ablation."
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "ENVI_CORRECTED_CROPPED"
    )
    parser.add_argument(
        "--pinn-root", type=Path, default=ROOT / "ENVI_PINN_DIFFUSION_AUGMENTED"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "predeployment_validation"
    )
    parser.add_argument("--synthetic-per-class", type=int, default=100)
    parser.add_argument("--selected-bands", type=int, default=40)
    parser.add_argument("--min-gap-nm", type=float, default=5.0)
    parser.add_argument("--radial-bins", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--cube-size", type=int, default=32)
    parser.add_argument("--cube-bands", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-size-3d", type=int, default=16)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    output = args.output_root
    cache_dir = output / "cache"
    csv_dir = output / "csv"
    model_dir = output / "models"
    for directory in (cache_dir, csv_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    real_train_records = discover_real_records(args.data_root, test=False)
    real_test_records = discover_real_records(args.data_root, test=True)
    synthetic_records = discover_augmented_records(
        args.pinn_root, "PINN-DDPM", args.synthetic_per_class
    )
    template = choose_template_meta(real_train_records)

    train_spectra_cache = cache_dir / "real_train_spectra.npz"
    if args.rebuild_cache or not train_spectra_cache.exists():
        spectra = []
        labels = []
        for index, record in enumerate(real_train_records, start=1):
            spectra.append(mean_nonzero_spectrum(read_fitted_cube(record.hdr_path, template)))
            labels.append(record.class_index)
            if index % 50 == 0 or index == len(real_train_records):
                print(f"[feature selection] {index}/{len(real_train_records)}")
        np.savez_compressed(
            train_spectra_cache,
            spectra=np.vstack(spectra),
            labels=np.asarray(labels, dtype=np.int64),
        )
    with np.load(train_spectra_cache) as cache:
        real_train_spectra = cache["spectra"].astype(np.float32)
        real_train_labels = cache["labels"].astype(np.int64)

    selected_idx, f_scores = select_wavelengths(
        real_train_spectra,
        real_train_labels,
        template.wavelengths,
        args.selected_bands,
        args.min_gap_nm,
    )
    selected_table = pd.DataFrame(
        {
            "rank": np.arange(1, len(selected_idx) + 1),
            "band_index": selected_idx,
            "wavelength_nm": template.wavelengths[selected_idx],
            "anova_f": f_scores[selected_idx],
        }
    )
    selected_table.to_csv(
        csv_dir / "selected_wavelengths_real_train_only.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_band_idx = int(selected_idx[0])
    print(
        "[wavelengths]",
        ", ".join(f"{template.wavelengths[index]:.0f}" for index in selected_idx),
    )

    cache_paths = {
        "real_train": cache_dir / "real_train_extracted.npz",
        "real_test": cache_dir / "real_test_extracted.npz",
        "pinn": cache_dir / "pinn_extracted.npz",
    }
    record_sets = {
        "real_train": real_train_records,
        "real_test": real_test_records,
        "pinn": synthetic_records,
    }
    extracted: dict[str, Extracted] = {}
    for label, records in record_sets.items():
        path = cache_paths[label]
        if args.rebuild_cache or not path.exists():
            data = extract_records(
                records,
                template,
                selected_idx,
                best_band_idx,
                args.radial_bins,
                args.image_size,
                args.cube_size,
                args.cube_bands,
                label,
            )
            save_extracted(path, data)
        extracted[label] = load_extracted(path)

    train_real = extracted["real_train"]
    train_synthetic = extracted["pinn"]
    test = extracted["real_test"]
    y_train_all = np.concatenate([train_real.y, train_synthetic.y])

    modality_inputs = {
        "Single best wavelength": (
            np.vstack([train_real.single, train_synthetic.single]),
            test.single,
        ),
        "Pseudo-RGB": (
            np.vstack([train_real.pseudo_features, train_synthetic.pseudo_features]),
            test.pseudo_features,
        ),
        "Full spectrum": (
            np.vstack([train_real.spectrum, train_synthetic.spectrum]),
            test.spectrum,
        ),
        "Selected wavelengths": (
            np.vstack(
                [
                    train_real.spectrum[:, selected_idx],
                    train_synthetic.spectrum[:, selected_idx],
                ]
            ),
            test.spectrum[:, selected_idx],
        ),
        "Spectral-spatial fusion": (
            np.vstack([train_real.fusion, train_synthetic.fusion]),
            test.fusion,
        ),
    }
    modality_rows: list[dict[str, float | str | int]] = []
    modality_predictions: dict[str, np.ndarray] = {}
    for name, (x_train, x_test) in modality_inputs.items():
        model = make_pipeline(
            StandardScaler(),
            SVC(C=10.0, gamma="scale", class_weight="balanced"),
        )
        start = time.perf_counter()
        model.fit(x_train, y_train_all)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = model.predict(x_test).astype(np.int64)
        inference_ms = (time.perf_counter() - start) * 1000.0 / len(test.y)
        modality_predictions[name] = prediction
        row: dict[str, float | str | int] = {
            "modality": name,
            "train_n": len(y_train_all),
            "test_n": len(test.y),
            "feature_dimension": x_train.shape[1],
            "train_seconds": train_seconds,
            "inference_ms_per_cube": inference_ms,
        }
        row.update(classification_metrics(test.y, prediction))
        row.update(
            bootstrap_ci(
                test.y, prediction, args.bootstrap_iterations, 9100 + len(modality_rows)
            )
        )
        modality_rows.append(row)
        pd.DataFrame(
            confusion_matrix(test.y, prediction, labels=np.arange(len(CLASS_LABELS))),
            index=CLASS_LABELS,
            columns=CLASS_LABELS,
        ).to_csv(
            csv_dir / f"confusion_modality_{name.lower().replace(' ', '_').replace('-', '_')}.csv",
            encoding="utf-8-sig",
        )
        print(f"[modality] {name}: {row['accuracy']:.4f}, {row['macro_f1']:.4f}")

    reference_name = max(modality_rows, key=lambda row: float(row["macro_f1"]))["modality"]
    modality_significance = []
    for name, prediction in modality_predictions.items():
        ref_only, comp_only, p_value = exact_mcnemar(
            test.y, modality_predictions[str(reference_name)], prediction
        )
        modality_significance.append(
            {
                "reference": reference_name,
                "comparator": name,
                "reference_only_correct": ref_only,
                "comparator_only_correct": comp_only,
                "discordant_n": ref_only + comp_only,
                "mcnemar_exact_p": p_value,
            }
        )
    pd.DataFrame(modality_rows).sort_values("macro_f1", ascending=False).to_csv(
        csv_dir / "hyperspectral_modality_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(modality_significance).to_csv(
        csv_dir / "hyperspectral_modality_mcnemar.csv",
        index=False,
        encoding="utf-8-sig",
    )

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=2026)
    real_subtrain_idx, real_val_idx = next(
        splitter.split(np.zeros(len(train_real.y)), train_real.y)
    )
    synthetic_offset = len(train_real.y)
    neural_train_indices = np.concatenate(
        [
            real_subtrain_idx,
            synthetic_offset + np.arange(len(train_synthetic.y)),
        ]
    )
    architecture_arrays = {
        "1D-CNN": (
            np.concatenate([train_real.spectrum, train_synthetic.spectrum])[:, None, :],
            test.spectrum[:, None, :],
        ),
        "2D-CNN": (
            np.concatenate([train_real.rgb, train_synthetic.rgb]).astype(np.float32),
            test.rgb.astype(np.float32),
        ),
        "3D-CNN": (
            np.concatenate([train_real.cube3d, train_synthetic.cube3d]).astype(np.float32),
            test.cube3d.astype(np.float32),
        ),
    }
    architecture_rows: list[dict[str, float | str | int]] = []
    ensemble_predictions: dict[str, np.ndarray] = {}
    device = torch.device(args.device)

    svm_model = make_pipeline(
        StandardScaler(),
        SVC(C=10.0, gamma="scale", class_weight="balanced", probability=False),
    )
    start = time.perf_counter()
    svm_model.fit(
        np.vstack([train_real.fusion, train_synthetic.fusion]),
        y_train_all,
    )
    svm_train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    svm_prediction = svm_model.predict(test.fusion).astype(np.int64)
    svm_inference_ms = (time.perf_counter() - start) * 1000.0 / len(test.y)
    ensemble_predictions["SVM-RBF"] = svm_prediction
    svm_row: dict[str, float | str | int] = {
        "architecture": "SVM-RBF",
        "family": "Machine learning",
        "seed": 2026,
        "train_n": len(y_train_all),
        "test_n": len(test.y),
        "parameters": int(svm_model.named_steps["svc"].support_vectors_.size),
        "best_epoch": 0,
        "validation_macro_f1": np.nan,
        "train_seconds": svm_train_seconds,
        "inference_ms_per_cube": svm_inference_ms,
    }
    svm_row.update(classification_metrics(test.y, svm_prediction))
    architecture_rows.append(svm_row)

    best_checkpoints: dict[str, tuple[float, dict[str, torch.Tensor], int]] = {}
    for architecture, (x_pool, x_test) in architecture_arrays.items():
        per_seed_predictions: list[np.ndarray] = []
        for seed in SEEDS:
            seed_everything(seed)
            start = time.perf_counter()
            model, prediction, metrics = train_neural(
                architecture=architecture,
                x_train=x_pool[neural_train_indices],
                y_train=y_train_all[neural_train_indices],
                x_val=x_pool[real_val_idx],
                y_val=train_real.y[real_val_idx],
                x_test=x_test,
                y_test=test.y,
                seed=seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size_3d
                if architecture == "3D-CNN"
                else args.batch_size,
                device=device,
            )
            train_seconds = time.perf_counter() - start
            per_seed_predictions.append(prediction)
            row: dict[str, float | str | int] = {
                "architecture": architecture,
                "family": architecture,
                "seed": seed,
                "train_n": len(neural_train_indices),
                "test_n": len(test.y),
                "train_seconds": train_seconds,
            }
            row.update(metrics)
            architecture_rows.append(row)
            checkpoint_path = model_dir / f"{architecture.lower().replace('-', '_')}_seed{seed}.pt"
            torch.save(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "class_labels": CLASS_LABELS,
                    "selected_wavelengths_nm": template.wavelengths[selected_idx].tolist(),
                    "state_dict": model.state_dict(),
                    "metrics": metrics,
                },
                checkpoint_path,
            )
            score = float(metrics["macro_f1"])
            current = best_checkpoints.get(architecture)
            if current is None or score > current[0]:
                best_checkpoints[architecture] = (
                    score,
                    deepcopy(model.state_dict()),
                    seed,
                )
            print(
                f"[architecture] {architecture} seed={seed}: "
                f"accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}"
            )
        ensemble_predictions[architecture] = majority_vote(per_seed_predictions)

    architecture_frame = pd.DataFrame(architecture_rows)
    architecture_frame.to_csv(
        csv_dir / "architecture_comparison_per_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_rows = []
    for architecture, group in architecture_frame.groupby("architecture", sort=False):
        ensemble = ensemble_predictions[architecture]
        row = {
            "architecture": architecture,
            "n_seeds": len(group),
            "accuracy_mean": group["accuracy"].mean(),
            "accuracy_sd": group["accuracy"].std(ddof=1) if len(group) > 1 else 0.0,
            "macro_f1_mean": group["macro_f1"].mean(),
            "macro_f1_sd": group["macro_f1"].std(ddof=1) if len(group) > 1 else 0.0,
            "mcc_mean": group["mcc"].mean(),
            "mcc_sd": group["mcc"].std(ddof=1) if len(group) > 1 else 0.0,
            "parameters_mean": group["parameters"].mean(),
            "inference_ms_per_cube_mean": group["inference_ms_per_cube"].mean(),
        }
        row.update(classification_metrics(test.y, ensemble))
        row.update(
            bootstrap_ci(
                test.y, ensemble, args.bootstrap_iterations, 12000 + len(summary_rows)
            )
        )
        summary_rows.append(row)
        pd.DataFrame(
            confusion_matrix(test.y, ensemble, labels=np.arange(len(CLASS_LABELS))),
            index=CLASS_LABELS,
            columns=CLASS_LABELS,
        ).to_csv(
            csv_dir
            / f"confusion_architecture_{architecture.lower().replace('-', '_')}.csv",
            encoding="utf-8-sig",
        )

    summary_frame = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary_frame.to_csv(
        csv_dir / "architecture_comparison_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_architecture = str(summary_frame.iloc[0]["architecture"])
    significance_rows = []
    for architecture, prediction in ensemble_predictions.items():
        ref_only, comp_only, p_value = exact_mcnemar(
            test.y, ensemble_predictions[best_architecture], prediction
        )
        significance_rows.append(
            {
                "reference": best_architecture,
                "comparator": architecture,
                "reference_only_correct": ref_only,
                "comparator_only_correct": comp_only,
                "discordant_n": ref_only + comp_only,
                "mcnemar_exact_p": p_value,
            }
        )
    pd.DataFrame(significance_rows).to_csv(
        csv_dir / "architecture_mcnemar.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_frame = pd.DataFrame(
        {
            "sample": test.names,
            "true_index": test.y,
            "true_label": [CLASS_LABELS[index] for index in test.y],
        }
    )
    for architecture, prediction in ensemble_predictions.items():
        prediction_frame[architecture] = [
            CLASS_LABELS[index] for index in prediction
        ]
    prediction_frame.to_csv(
        csv_dir / "architecture_test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    run_config = {
        "data_root": str(args.data_root.resolve()),
        "pinn_root": str(args.pinn_root.resolve()),
        "real_train_per_class": 70,
        "independent_test_per_class": 30,
        "synthetic_per_class": args.synthetic_per_class,
        "feature_selection": "ANOVA on real training cubes only",
        "selected_band_count": args.selected_bands,
        "selected_wavelengths_nm": template.wavelengths[selected_idx].tolist(),
        "single_best_wavelength_nm": float(template.wavelengths[best_band_idx]),
        "neural_seeds": SEEDS,
        "device": str(device),
        "torch": torch.__version__,
        "best_architecture_by_ensemble_macro_f1": best_architecture,
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n[modality summary]")
    print(pd.DataFrame(modality_rows).sort_values("macro_f1", ascending=False).to_string(index=False))
    print("\n[architecture summary]")
    print(summary_frame.to_string(index=False))
    print(f"\n[best architecture] {best_architecture}")
    print(f"[outputs] {output}")


if __name__ == "__main__":
    main()
