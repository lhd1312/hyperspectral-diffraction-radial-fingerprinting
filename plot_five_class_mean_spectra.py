"""Draw five class means from the archived real-sample ROI spectra."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "paper_real_sample_spectra_cv" / "source_data"
SOURCE = INPUT_DIR / "sample_level_spatial_mean_spectra.csv"
OUTPUT = ROOT / "paper_real_sample_spectra_cv" / "mean_spectra"
LABELS = {"CMB": "Fg", "DQB": "Uv", "DWB": "Po", "WQ": "PVC", "YMXB": "Pp"}
COLORS = {"Fg": "#0072B2", "Uv": "#E69F00", "Po": "#009E73", "PVC": "#CC79A7", "Pp": "#D55E00"}


def load_means() -> tuple[pd.DataFrame, dict]:
    samples = pd.read_csv(SOURCE)
    key = ["class", "subset", "sample", "wavelength_nm"]
    if samples.duplicated(key).any():
        raise ValueError("Duplicate sample/wavelength records")
    if set(samples["class"]) != set(LABELS) or set(samples["subset"]) != {"train", "test"}:
        raise ValueError("Unexpected sample classes or subsets")
    if not np.isfinite(samples[["wavelength_nm", "roi_mean_intensity"]].to_numpy()).all():
        raise ValueError("Non-finite wavelengths or intensities")

    waves = np.sort(samples.wavelength_nm.unique())
    means, counts = {}, {}
    for internal, label in LABELS.items():
        rows = samples.loc[samples["class"].eq(internal)]
        matrix = rows.pivot(index=["subset", "sample"], columns="wavelength_nm", values="roi_mean_intensity")
        matrix = matrix.reindex(columns=waves)
        if matrix.isna().any().any():
            raise ValueError(f"Incomplete wavelength coverage: {internal}")
        subset_counts = matrix.groupby(level="subset").size().to_dict()
        if subset_counts != {"test": 30, "train": 70}:
            raise ValueError(f"Unexpected sample counts: {internal}: {subset_counts}")
        for subset, sample in matrix.index:
            folder = f"DATA{internal}" + ("_test" if subset == "test" else "")
            header = ROOT / "ENVI_CORRECTED_CROPPED" / folder / f"{sample}.hdr"
            if not header.is_file():
                raise FileNotFoundError(header)
        means[label] = matrix.mean(axis=0).to_numpy()
        counts[label] = {"total": len(matrix), **subset_counts}

    output = pd.DataFrame(means, index=pd.Index(waves, name="wavelength_nm"))
    reference = pd.read_csv(INPUT_DIR / "class_spectral_summary_and_cv.csv")
    for internal, label in LABELS.items():
        expected = reference.loc[reference["class"].eq(internal)].set_index("wavelength_nm").loc[waves, "mean_roi_intensity"]
        np.testing.assert_allclose(output[label], expected, rtol=1e-12, atol=1e-12)
    return output, counts


def main() -> None:
    # Descriptive comparison only: equal-weight ROI means; no fitting or selection.
    means, counts = load_means()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "legend.fontsize": 10,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.75, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(105 / 25.4, 72 / 25.4))
    ax = fig.add_axes([0.18, 0.19, 0.79, 0.67])
    for label in means:
        ax.plot(means.index, means[label], color=COLORS[label], lw=1.25, label=label)
    ax.set(xlim=(400, 800), ylim=(0, 110), xlabel="Wavelength (nm)", ylabel="Intensity (a.u.)")
    ax.set_xticks([400, 500, 600, 700, 800])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(length=3, width=0.75, pad=4)
    fig.legend(*ax.get_legend_handles_labels(), ncol=5, loc="center", bbox_to_anchor=(0.565, 0.94),
               handlelength=1.15, handletextpad=0.35, columnspacing=0.85, borderpad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        assert text.get_fontsize() == 10
        box = text.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        if box.x0 < 0 or box.y0 < 0 or box.x1 > fig.bbox.x1 or box.y1 > fig.bbox.y1:
            raise ValueError(f"Text outside canvas: {text.get_text()}")
        boxes.append((text.get_text(), box))
    for i, (first, a) in enumerate(boxes):
        for second, b in boxes[i + 1:]:
            if a.overlaps(b):
                raise ValueError(f"Overlapping labels: {first}, {second}")
    assert len(ax.lines) == 5
    for line, label in zip(ax.lines, means.columns):
        np.testing.assert_array_equal(line.get_ydata(), means[label].to_numpy())

    stem = OUTPUT / "five_class_mean_spectra"
    for extension in ("svg", "pdf", "png", "tiff"):
        options = {"dpi": 600} if extension in ("png", "tiff") else {}
        if extension == "tiff":
            options["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(stem.with_suffix(f".{extension}"), **options)
    plt.close(fig)
    svg = ET.parse(stem.with_suffix(".svg")).getroot()
    assert not list(svg.iter("{http://www.w3.org/2000/svg}image"))
    assert len(list(svg.iter("{http://www.w3.org/2000/svg}text"))) >= 5
    means.to_csv(OUTPUT / "five_class_mean_spectra.csv")
    manifest = {
        "purpose": "Compare the measured mean corrected ROI spectra of five particle classes; not evidence of classifier performance.",
        "archetype": "single-axis quantitative comparison", "backend": "Python/matplotlib",
        "input": "archived per-sample spatial-mean spectra from real corrected ENVI cubes",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "class_mapping": LABELS, "colors": COLORS, "sample_counts": counts,
        "mean_definition": "At each wavelength, arithmetic mean of the spatial-mean spectra, with equal weight for each ROI cube.",
        "bands": len(means), "wavelength_range_nm": [float(means.index.min()), float(means.index.max())],
        "additional_smoothing": False, "additional_normalization": False,
        "synthetic_samples": False, "uncertainty_displayed": False,
        "test_data_use": "Post hoc descriptive display only, not training, feature selection or model tuning.",
        "pixel_scope": "Full stored ROI, including any masked pixels retained in the preprocessed cube; inherited from the original extraction.",
        "intensity_label": "Corrected ROI-mean intensity in arbitrary units, not reflectance.",
        "canvas_mm": [105, 72], "font_pt": 10, "raster_dpi": 600,
        "qa": {"exactly_five_curves": True, "matches_archived_summary": True,
               "source_headers_exist": True, "text_bounds_and_overlap": "passed", "vector_text_editable": True},
    }
    (OUTPUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (OUTPUT / "CAPTION.txt").write_text(
        "Mean corrected diffraction spectra of five particle classes. Each curve is the arithmetic mean of "
        "100 real ROI spectra (70 training and 30 held-out test ROI cubes); each ROI spectrum was obtained "
        "by spatially averaging the stored cube at each wavelength. Curves cover 369 bands from 411 to 779 nm. "
        "Colors identify Fg, Uv, Po, PVC and Pp. No additional smoothing or normalization was applied, "
        "and no synthetic samples are included. Pooling the held-out samples is restricted to this "
        "post hoc descriptive visualization. Curves show means only, without uncertainty bands.\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(stem), "samples": counts, "bands": len(means), "qa": "passed"}, indent=2))


if __name__ == "__main__":
    main()
