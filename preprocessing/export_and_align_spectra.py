from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "SPECTRA_CSV"
COMMON_WAVELENGTHS = np.arange(400.0, 801.0, 1.0)
MAX_SHIFT_BANDS = 30
MANUAL_CATEGORY_SHIFTS = {
    "cmb": 0,
    "dwb": -5,
    "dqb": -12,
    "wq": -18,
    "ymxb": -4,
}

CATEGORIES = {
    "cmb": ROOT / "DATACMB",
    "dwb": ROOT / "DATADWB",
    "dqb": ROOT / "DATADQB",
    "wq": ROOT / "DATAWQ",
    "ymxb": ROOT / "DATAYMXB",
}

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
class EnviMeta:
    samples: int
    lines: int
    bands: int
    data_type: int
    wavelengths: np.ndarray


@dataclass
class SpectrumSet:
    category: str
    sample_names: list[str]
    wavelengths: np.ndarray
    spectra: np.ndarray
    common_spectra: np.ndarray
    shifts: np.ndarray | None = None
    aligned: np.ndarray | None = None


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 10**9, path.stem)


def parse_int_field(text: str, field: str) -> int:
    match = re.search(rf"^{re.escape(field)}\s*=\s*(\d+)", text, flags=re.MULTILINE | re.IGNORECASE)
    if not match:
        raise ValueError(f"Missing ENVI header field: {field}")
    return int(match.group(1))


def parse_wavelengths(text: str, bands: int) -> np.ndarray:
    match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return np.linspace(400.0, 800.0, bands)
    values = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
    if len(values) != bands:
        return np.linspace(400.0, 800.0, bands)
    return np.asarray(values, dtype=np.float64)


def read_meta(hdr_path: Path) -> EnviMeta:
    text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    bands = parse_int_field(text, "bands")
    return EnviMeta(
        samples=parse_int_field(text, "samples"),
        lines=parse_int_field(text, "lines"),
        bands=bands,
        data_type=parse_int_field(text, "data type"),
        wavelengths=parse_wavelengths(text, bands),
    )


def read_mean_spectrum(hdr_path: Path) -> tuple[np.ndarray, np.ndarray]:
    meta = read_meta(hdr_path)
    dtype = DTYPE_MAP.get(meta.data_type)
    if dtype is None:
        raise ValueError(f"Unsupported ENVI data type {meta.data_type}: {hdr_path}")

    raw_path = hdr_path.with_suffix(".raw")
    expected = meta.bands * meta.lines * meta.samples
    data = np.fromfile(raw_path, dtype=dtype, count=expected)
    if data.size != expected:
        raise ValueError(f"Raw size mismatch for {raw_path}: {data.size} values, expected {expected}")

    cube = data.reshape((meta.bands, meta.lines, meta.samples)).astype(np.float64)
    flat = cube.reshape(meta.bands, -1)
    nonzero_count = np.count_nonzero(flat, axis=1)
    summed = flat.sum(axis=1)
    full_mean = flat.mean(axis=1)
    spectrum = np.divide(
        summed,
        nonzero_count,
        out=full_mean.copy(),
        where=nonzero_count > 0,
    )
    return meta.wavelengths, spectrum


def write_csv(path: Path, wavelengths: np.ndarray, sample_names: list[str], spectra: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["wavelength", *sample_names])
        for row_idx, wavelength in enumerate(wavelengths):
            writer.writerow([f"{wavelength:.4f}", *[f"{value:.8g}" for value in spectra[:, row_idx]]])


def moving_average(values: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def zscore(values: np.ndarray) -> np.ndarray:
    centered = values - np.nanmean(values)
    std = np.nanstd(centered)
    if not np.isfinite(std) or std == 0:
        return centered
    return centered / std


def best_shift(reference: np.ndarray, sample: np.ndarray, max_shift: int) -> int:
    reference_feature = zscore(np.gradient(moving_average(reference)))
    sample_feature = zscore(np.gradient(moving_average(sample)))

    best_score = -np.inf
    best_lag = 0
    for lag in range(-max_shift, max_shift + 1):
        if lag > 0:
            ref_slice = reference_feature[lag:]
            sample_slice = sample_feature[:-lag]
        elif lag < 0:
            ref_slice = reference_feature[:lag]
            sample_slice = sample_feature[-lag:]
        else:
            ref_slice = reference_feature
            sample_slice = sample_feature

        valid = np.isfinite(ref_slice) & np.isfinite(sample_slice)
        if valid.sum() < 20:
            continue
        score = float(np.dot(ref_slice[valid], sample_slice[valid]) / valid.sum())
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag


def apply_shift(sample: np.ndarray, lag: int) -> np.ndarray:
    shifted = np.full_like(sample, np.nan, dtype=np.float64)
    if lag > 0:
        shifted[lag:] = sample[:-lag]
    elif lag < 0:
        shifted[:lag] = sample[-lag:]
    else:
        shifted[:] = sample
    return shifted


def load_category(category: str, folder: Path) -> SpectrumSet:
    hdr_files = sorted(folder.glob("*.hdr"), key=natural_key)[:100]
    if len(hdr_files) != 100:
        raise ValueError(f"{folder.name} should contain 100 hdr files, found {len(hdr_files)}")

    sample_names: list[str] = []
    wavelengths: np.ndarray | None = None
    spectra: list[np.ndarray] = []
    common_spectra: list[np.ndarray] = []
    for hdr_path in hdr_files:
        sample_wavelengths, spectrum = read_mean_spectrum(hdr_path)
        sample_names.append(hdr_path.stem)
        if wavelengths is None:
            wavelengths = sample_wavelengths
        elif len(sample_wavelengths) != len(wavelengths) or not np.allclose(sample_wavelengths, wavelengths):
            wavelengths = None
        spectra.append(spectrum)
        common_spectra.append(np.interp(COMMON_WAVELENGTHS, sample_wavelengths, spectrum))

    if wavelengths is None:
        wavelengths = COMMON_WAVELENGTHS
        spectra_array = np.vstack(common_spectra)
    else:
        spectra_array = np.vstack(spectra)

    return SpectrumSet(
        category=category,
        sample_names=sample_names,
        wavelengths=wavelengths,
        spectra=spectra_array,
        common_spectra=np.vstack(common_spectra),
    )


def align_sets(sets: list[SpectrumSet]) -> np.ndarray:
    all_aligned: list[np.ndarray] = []
    for item in sets:
        reference = np.nanmedian(item.common_spectra, axis=0)
        shifts = np.asarray(
            [best_shift(reference, spectrum, MAX_SHIFT_BANDS) for spectrum in item.common_spectra],
            dtype=int,
        )
        item.shifts = shifts
        item.aligned = np.vstack(
            [apply_shift(spectrum, int(shift)) for spectrum, shift in zip(item.common_spectra, shifts)]
        )
        manual_shift = MANUAL_CATEGORY_SHIFTS.get(item.category, 0)
        if manual_shift:
            item.aligned = np.vstack([apply_shift(spectrum, manual_shift) for spectrum in item.aligned])
        all_aligned.append(item.aligned)

    valid_rows = np.all(np.isfinite(np.vstack(all_aligned)), axis=0)
    return valid_rows


def write_shift_report(sets: list[SpectrumSet]) -> None:
    report_path = OUTPUT_DIR / "band_shift_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "sample", "shift_bands"])
        for item in sets:
            assert item.shifts is not None
            manual_shift = MANUAL_CATEGORY_SHIFTS.get(item.category, 0)
            for sample_name, shift in zip(item.sample_names, item.shifts):
                writer.writerow([item.category, sample_name, int(shift) + manual_shift])


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sets = [load_category(category, folder) for category, folder in CATEGORIES.items()]

    raw_dir = OUTPUT_DIR / "raw"
    for item in sets:
        write_csv(raw_dir / f"{item.category}_spectra_raw.csv", item.wavelengths, item.sample_names, item.spectra)

    valid_rows = align_sets(sets)
    cropped_wavelengths = COMMON_WAVELENGTHS[valid_rows]
    corrected_dir = OUTPUT_DIR / "corrected_cropped"
    for item in sets:
        assert item.aligned is not None
        write_csv(
            corrected_dir / f"{item.category}_spectra_corrected_cropped.csv",
            cropped_wavelengths,
            item.sample_names,
            item.aligned[:, valid_rows],
        )

    write_shift_report(sets)

    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Raw CSV files: {raw_dir}")
    print(f"Corrected cropped CSV files: {corrected_dir}")
    print(
        "Cropped wavelength range: "
        f"{cropped_wavelengths[0]:.0f}-{cropped_wavelengths[-1]:.0f} nm "
        f"({len(cropped_wavelengths)} bands)"
    )
    for item in sets:
        assert item.shifts is not None
        total_shifts = item.shifts + MANUAL_CATEGORY_SHIFTS.get(item.category, 0)
        print(
            f"{item.category}: total shift min={total_shifts.min()}, "
            f"max={total_shifts.max()}, median={np.median(total_shifts):.1f}"
        )


if __name__ == "__main__":
    main()
