"""Plot class spectra and sample-level CV from all real ENVI cubes.

Each cube is spatially averaged first, yielding one 369-band spectrum per
real sample. Training and untouched real-test cubes are pooled only for this
descriptive visualization. Synthetic/augmented cubes are never read.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "ENVI_CORRECTED_CROPPED"
DEFAULT_OUTPUT_ROOT = ROOT / "paper_real_sample_spectra_cv"

CLASS_ORDER = ["CMB", "DQB", "DWB", "WQ", "YMXB"]
CLASS_COLORS = {
    "CMB": "#367C9F",
    "DQB": "#D4743B",
    "DWB": "#43916E",
    "WQ": "#8667A8",
    "YMXB": "#B95B62",
}
CLASS_FOLDERS = {
    "CMB": ("DATACMB", "DATACMB_test"),
    "DQB": ("DATADQB", "DATADQB_test"),
    "DWB": ("DATADWB", "DATADWB_test"),
    "WQ": ("DATAWQ", "DATAWQ_test"),
    "YMXB": ("DATAYMXB", "DATAYMXB_test"),
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
    interleave: str
    byte_order: int
    wavelengths: np.ndarray


def parse_int(text: str, field: str, default: int | None = None) -> int:
    match = re.search(
        rf"^{re.escape(field)}\s*=\s*(-?\d+)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if match:
        return int(match.group(1))
    if default is not None:
        return default
    raise ValueError(f"Missing ENVI field: {field}")


def parse_str(text: str, field: str, default: str = "") -> str:
    match = re.search(
        rf"^{re.escape(field)}\s*=\s*(.+)$",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip().strip("{}").strip() if match else default


def parse_wavelengths(text: str, bands: int) -> np.ndarray:
    match = re.search(
        r"wavelength\s*=\s*\{([^}]*)\}",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        raise ValueError("Missing ENVI wavelength list")
    values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
    if len(values) != bands:
        raise ValueError(f"Expected {bands} wavelengths, found {len(values)}")
    return np.asarray(values, dtype=np.float64)


def read_envi_cube(header: Path) -> tuple[EnviMeta, np.ndarray]:
    text = header.read_text(encoding="utf-8", errors="ignore")
    bands = parse_int(text, "bands")
    meta = EnviMeta(
        samples=parse_int(text, "samples"),
        lines=parse_int(text, "lines"),
        bands=bands,
        data_type=parse_int(text, "data type"),
        interleave=parse_str(text, "interleave", "bsq").lower(),
        byte_order=parse_int(text, "byte order", 0),
        wavelengths=parse_wavelengths(text, bands),
    )
    if meta.data_type not in DTYPE_MAP:
        raise ValueError(f"Unsupported ENVI data type: {meta.data_type}")
    dtype = np.dtype(DTYPE_MAP[meta.data_type]).newbyteorder(">" if meta.byte_order == 1 else "<")
    expected = meta.bands * meta.lines * meta.samples
    data = np.fromfile(header.with_suffix(".raw"), dtype=dtype, count=expected)
    if data.size != expected:
        raise ValueError(f"Raw size mismatch: {header.with_suffix('.raw')}")
    if meta.interleave == "bsq":
        cube = data.reshape(meta.bands, meta.lines, meta.samples)
    elif meta.interleave == "bil":
        cube = data.reshape(meta.lines, meta.bands, meta.samples).transpose(1, 0, 2)
    elif meta.interleave == "bip":
        cube = data.reshape(meta.lines, meta.samples, meta.bands).transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported ENVI interleave: {meta.interleave}")
    cube = cube.astype(np.float32, copy=False)
    cube[~np.isfinite(cube)] = 0.0
    return meta, cube

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.75,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 10**9, path.stem.casefold())


def save_figure(fig: mpl.figure.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(
        base_path.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def categorical_sample_colors(n: int) -> np.ndarray:
    """Return a deterministic multi-hue color for every sample curve."""
    palette_names = (
        "tab20",
        "tab20b",
        "tab20c",
        "Paired",
        "Set1",
        "Set2",
        "Set3",
        "Dark2",
        "Accent",
    )
    colors: list[tuple[float, float, float, float]] = []
    for palette_name in palette_names:
        cmap = mpl.colormaps[palette_name]
        palette = getattr(cmap, "colors", cmap(np.linspace(0.0, 1.0, cmap.N)))
        colors.extend(mpl.colors.to_rgba(color) for color in palette)
    unique_colors: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for color in colors:
        key = tuple(float(value) for value in np.round(color, 6))
        if key not in seen:
            seen.add(key)
            unique_colors.append(color)
    rng = np.random.default_rng(20260830)
    shuffled = np.asarray(unique_colors)[rng.permutation(len(unique_colors))]
    if len(shuffled) < n:
        extra = mpl.colormaps["turbo"](np.linspace(0.02, 0.98, n - len(shuffled)))
        shuffled = np.vstack([shuffled, extra])
    return shuffled[:n]


def load_real_sample_spectra(
    data_root: Path,
) -> tuple[np.ndarray, dict[str, dict[str, object]]]:
    reference_wavelengths: np.ndarray | None = None
    results: dict[str, dict[str, object]] = {}
    total_expected = sum(
        len(list((data_root / folder).glob("*.hdr")))
        for folders in CLASS_FOLDERS.values()
        for folder in folders
    )
    completed = 0

    for label in CLASS_ORDER:
        spectra: list[np.ndarray] = []
        sample_names: list[str] = []
        subsets: list[str] = []
        spatial_shapes: list[tuple[int, int]] = []
        for subset, folder_name in zip(("train", "test"), CLASS_FOLDERS[label]):
            folder = data_root / folder_name
            headers = sorted(folder.glob("*.hdr"), key=natural_key)
            if not headers:
                raise FileNotFoundError(f"No ENVI headers found in {folder}")
            for header in headers:
                meta, cube = read_envi_cube(header)
                wavelengths = meta.wavelengths.astype(np.float64)
                if reference_wavelengths is None:
                    reference_wavelengths = wavelengths
                elif not np.allclose(wavelengths, reference_wavelengths):
                    raise ValueError(f"Wavelength mismatch: {header}")
                spectra.append(cube.mean(axis=(1, 2), dtype=np.float64))
                sample_names.append(header.stem)
                subsets.append(subset)
                spatial_shapes.append((meta.lines, meta.samples))
                completed += 1
                if completed % 25 == 0 or completed == total_expected:
                    print(f"[spectra] {completed}/{total_expected} real cubes", flush=True)
        matrix = np.stack(spectra).astype(np.float64)
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"Non-finite sample spectrum found for {label}")
        mean = matrix.mean(axis=0)
        sd = matrix.std(axis=0, ddof=1)
        cv = 100.0 * sd / np.maximum(np.abs(mean), np.finfo(float).eps)
        results[label] = {
            "spectra": matrix,
            "sample_names": sample_names,
            "subsets": subsets,
            "spatial_shapes": spatial_shapes,
            "mean": mean,
            "sd": sd,
            "cv": cv,
        }

    if reference_wavelengths is None:
        raise RuntimeError("No real ENVI cube was loaded")
    return reference_wavelengths, results


def plot_individual_spectra(
    wavelengths: np.ndarray,
    results: dict[str, dict[str, object]],
    output_root: Path,
) -> None:
    global_upper = max(
        float(np.max(np.asarray(results[label]["spectra"]))) for label in CLASS_ORDER
    )
    y_upper = global_upper * 1.06
    tick_start = int(np.ceil(wavelengths.min() / 60.0) * 60)
    x_ticks = np.arange(tick_start, wavelengths.max() + 1, 60)

    for label in CLASS_ORDER:
        matrix = np.asarray(results[label]["spectra"])
        n = matrix.shape[0]
        curve_colors = categorical_sample_colors(n)
        fig, axis = plt.subplots(figsize=(89 / 25.4, 59 / 25.4))
        for sample_index in range(n):
            axis.plot(
                wavelengths,
                matrix[sample_index],
                color=curve_colors[sample_index],
                lw=0.38,
                alpha=0.78,
                rasterized=False,
            )
        axis.set_xlim(float(wavelengths.min()), float(wavelengths.max()))
        axis.set_ylim(0.0, y_upper)
        axis.set_xticks(x_ticks)
        axis.set_xlabel("Wavelength (nm)")
        axis.set_ylabel("ROI-mean corrected intensity (a.u.)")
        axis.grid(axis="y", color="#E2E5E8", lw=0.45, zorder=0)
        axis.text(
            0.025,
            0.945,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=CLASS_COLORS[label],
            fontsize=8.2,
            fontweight="bold",
        )
        axis.text(
            0.975,
            0.945,
            f"n = {n} individual real-sample spectra",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color="#60676D",
            fontsize=5.8,
        )
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.22, top=0.975)
        save_figure(fig, output_root / "individual" / f"spectral_curve_{label}")
        plt.close(fig)


def plot_combined_cv(
    wavelengths: np.ndarray,
    results: dict[str, dict[str, object]],
    output_root: Path,
) -> None:
    distributions = [np.asarray(results[label]["cv"]) for label in CLASS_ORDER]
    all_cv = np.concatenate(distributions)
    y_upper = float(np.max(all_cv)) * 1.13
    positions = np.arange(1, len(CLASS_ORDER) + 1)
    fig, axis = plt.subplots(figsize=(89 / 25.4, 62 / 25.4))
    violins = axis.violinplot(
        distributions,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.24,
    )
    for body, label in zip(violins["bodies"], CLASS_ORDER):
        body.set_facecolor(CLASS_COLORS[label])
        body.set_edgecolor(CLASS_COLORS[label])
        body.set_alpha(0.58)
        body.set_linewidth(0.75)

    for position, label, values in zip(positions, CLASS_ORDER, distributions):
        q05, q25, median, q75, q95 = np.percentile(values, [5, 25, 50, 75, 95])
        color = CLASS_COLORS[label]
        axis.vlines(position, q05, q95, color=color, lw=0.9, zorder=3)
        axis.vlines(position, q25, q75, color=color, lw=4.2, zorder=4)
        axis.scatter(
            position,
            median,
            s=16,
            facecolor="white",
            edgecolor=color,
            linewidth=0.75,
            zorder=5,
        )
        axis.text(
            position,
            median + 0.23,
            f"{median:.2f}%",
            ha="center",
            va="bottom",
            fontsize=5.2,
            color="#303438",
            zorder=6,
        )

    axis.set_xlim(0.45, len(CLASS_ORDER) + 0.55)
    axis.set_ylim(0.0, y_upper)
    axis.set_xticks(positions, CLASS_ORDER)
    axis.set_ylabel("Between-sample CV across wavelengths (%)")
    axis.grid(axis="y", color="#E2E5E8", lw=0.45, zorder=0)
    for tick, label in zip(axis.get_xticklabels(), CLASS_ORDER):
        tick.set_color(CLASS_COLORS[label])
        tick.set_fontweight("bold")
    axis.text(
        0.995,
        0.985,
        "369 bands per class; each CV uses n = 100 samples",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color="#60676D",
        fontsize=5.7,
    )
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.17, top=0.975)
    save_figure(fig, output_root / "spectral_cv_five_classes")
    plt.close(fig)


def write_source_data(
    wavelengths: np.ndarray,
    results: dict[str, dict[str, object]],
    output_root: Path,
) -> None:
    source_dir = output_root / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)

    with (source_dir / "sample_level_spatial_mean_spectra.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "subset", "sample", "wavelength_nm", "roi_mean_intensity"])
        for label in CLASS_ORDER:
            item = results[label]
            for sample_name, subset, spectrum in zip(
                item["sample_names"], item["subsets"], np.asarray(item["spectra"])
            ):
                for wavelength, intensity in zip(wavelengths, spectrum):
                    writer.writerow([label, subset, sample_name, wavelength, intensity])

    with (source_dir / "class_spectral_summary_and_cv.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "class",
                "n_real_cubes",
                "wavelength_nm",
                "mean_roi_intensity",
                "sample_sd",
                "sample_cv_percent",
            ]
        )
        for label in CLASS_ORDER:
            item = results[label]
            n = np.asarray(item["spectra"]).shape[0]
            for wavelength, mean, sd, cv in zip(
                wavelengths, item["mean"], item["sd"], item["cv"]
            ):
                writer.writerow([label, n, wavelength, mean, sd, cv])

    manifest = {
        "input_scope": "all 500 real corrected ENVI cubes; no synthetic samples",
        "per_class": {label: int(np.asarray(results[label]["spectra"]).shape[0]) for label in CLASS_ORDER},
        "train_per_class": 70,
        "test_per_class": 30,
        "bands": int(len(wavelengths)),
        "wavelength_range_nm": [float(wavelengths.min()), float(wavelengths.max())],
        "sample_spectrum_definition": "spatial mean over all pixels in each real cube at each wavelength",
        "individual_curve": "all 100 individual sample spectra; one categorical color per sample with no numeric meaning",
        "cv_definition": "100 × sample SD / absolute class mean at each wavelength",
        "cv_visualization": "violin distribution of 369 wavelength-specific CV values; dot=median, thick bar=IQR, whisker=5th-95th percentile",
        "use_of_test_set": "post hoc descriptive visualization only; no fitting, selection, or tuning",
    }
    (source_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Real-sample spectra and CV",
        "",
        "All figures use the 500 real corrected ENVI cubes only (100 per class: 70 training + 30 untouched real test).",
        "Each cube is spatially averaged first to obtain one 369-band sample spectrum.",
        "Individual figures show all 100 sample spectra with one categorical color per sample.",
        "The combined violin figure summarizes the 369 wavelength-specific CV values per class.",
        "In each violin, the white dot is the median, the thick bar is the IQR, and the thin whisker is the 5th-95th percentile.",
        "Pooling the test set is restricted to post hoc descriptive visualization and does not affect model fitting or wavelength selection.",
    ]
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    wavelengths, results = load_real_sample_spectra(args.data_root)
    plot_individual_spectra(wavelengths, results, output_root)
    plot_combined_cv(wavelengths, results, output_root)
    write_source_data(wavelengths, results, output_root)
    for label in CLASS_ORDER:
        cv = np.asarray(results[label]["cv"])
        print(
            f"[{label}] n={np.asarray(results[label]['spectra']).shape[0]}, "
            f"median CV={np.median(cv):.3f}%, range={cv.min():.3f}-{cv.max():.3f}%"
        )
    print(output_root)


if __name__ == "__main__":
    main()
