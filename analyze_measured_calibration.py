from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "calibration"
SEED = 20260721
BOOTSTRAP_ITERATIONS = 10_000
FINAL_GRADIENT_MAP = {
    "C0": "G0",
    "C1": "G1",
    "C3": "G2",
    "C5": "G3",
    "C7": "G4",
    "C8": "G5",
}


def ols_fit(x: np.ndarray, y: np.ndarray) -> dict[str, float | np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if x.ndim != 1 or y.shape != x.shape or n < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OLS needs at least three finite, paired observations")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("OLS requires variable counts and reference concentrations")
    x_bar = float(x.mean())
    y_bar = float(y.mean())
    sxx = float(np.sum((x - x_bar) ** 2))
    sxy = float(np.sum((x - x_bar) * (y - y_bar)))
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    fitted = intercept + slope * x
    residuals = y - fitted
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - y_bar) ** 2))
    r2 = 1.0 - sse / sst
    residual_se = math.sqrt(sse / (n - 2))
    slope_se = residual_se / math.sqrt(sxx)
    intercept_se = residual_se * math.sqrt(1.0 / n + x_bar**2 / sxx)
    t_crit = float(stats.t.ppf(0.975, df=n - 2))
    return {
        "n": n,
        "x_bar": x_bar,
        "y_bar": y_bar,
        "sxx": sxx,
        "slope": slope,
        "intercept": intercept,
        "fitted": fitted,
        "residuals": residuals,
        "sse": sse,
        "sst": sst,
        "r2": r2,
        "adjusted_r2": 1.0 - (1.0 - r2) * (n - 1) / (n - 2),
        "rmse": math.sqrt(float(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "residual_se": residual_se,
        "slope_se": slope_se,
        "intercept_se": intercept_se,
        "slope_ci_low": slope - t_crit * slope_se,
        "slope_ci_high": slope + t_crit * slope_se,
        "intercept_ci_low": intercept - t_crit * intercept_se,
        "intercept_ci_high": intercept + t_crit * intercept_se,
        "t_crit": t_crit,
    }


def leave_one_out_predictions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    predictions = np.empty_like(y, dtype=float)
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        fit = ols_fit(x[keep], y[keep])
        predictions[i] = fit["intercept"] + fit["slope"] * x[i]
    return predictions


def stratified_bootstrap(
    prep: pd.DataFrame, iterations: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = [group.reset_index(drop=True) for _, group in prep.groupby("gradient_id")]
    slopes = np.empty(iterations, dtype=float)
    intercepts = np.empty(iterations, dtype=float)
    for i in range(iterations):
        sampled = [
            group.iloc[rng.integers(0, len(group), size=len(group))]
            for group in groups
        ]
        boot = pd.concat(sampled, ignore_index=True)
        fit = ols_fit(
            boot["mean_ring_count"].to_numpy(),
            boot["mean_reference_concentration_particles_ml"].to_numpy(),
        )
        slopes[i] = fit["slope"]
        intercepts[i] = fit["intercept"]
    return slopes, intercepts


def save_publication_figure(
    prep: pd.DataFrame,
    gradient: pd.DataFrame,
    fit: dict[str, float | np.ndarray],
    curve: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.linewidth": 0.75,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "legend.frameon": False,
            "lines.linewidth": 1.1,
        }
    )

    palette = {
        "G0": "#6B7280",
        "G1": "#3B82A0",
        "G2": "#2A9D8F",
        "G3": "#D69E2E",
        "G4": "#D97757",
        "G5": "#9C4F72",
    }

    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 105 * mm), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.62, 1.0],
        height_ratios=[1, 1],
        left=0.075,
        right=0.985,
        bottom=0.13,
        top=0.965,
        wspace=0.34,
        hspace=0.48,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    x_scale = 1.0
    y_scale = 1_000.0

    ax_a.fill_between(
        curve["mean_ring_count"] / x_scale,
        curve["prediction_low"] / y_scale,
        curve["prediction_high"] / y_scale,
        color="#DCE8EC",
        alpha=0.55,
        linewidth=0,
        label="95% prediction interval",
        zorder=1,
    )
    ax_a.fill_between(
        curve["mean_ring_count"] / x_scale,
        curve["confidence_low"] / y_scale,
        curve["confidence_high"] / y_scale,
        color="#8FC1CC",
        alpha=0.55,
        linewidth=0,
        label="95% confidence interval",
        zorder=2,
    )
    ax_a.plot(
        curve["mean_ring_count"] / x_scale,
        curve["fitted_concentration_particles_ml"] / y_scale,
        color="#174A5B",
        linewidth=1.5,
        label="Ordinary least squares",
        zorder=3,
    )

    for gradient_id, group in prep.groupby("gradient_id", sort=False):
        ax_a.scatter(
            group["mean_ring_count"] / x_scale,
            group["mean_reference_concentration_particles_ml"] / y_scale,
            s=28,
            facecolor=palette[gradient_id],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.95,
            label=gradient_id,
            zorder=5,
        )

    for _, row in gradient.iterrows():
        ax_a.errorbar(
            row["mean_ring_count"] / x_scale,
            row["mean_reference_concentration_particles_ml"] / y_scale,
            xerr=row["sd_ring_count"] / x_scale,
            yerr=row["sd_reference_concentration_particles_ml"] / y_scale,
            fmt="D",
            markersize=3.8,
            markerfacecolor=palette[row["gradient_id"]],
            markeredgecolor="#24333A",
            markeredgewidth=0.45,
            ecolor="#52636A",
            elinewidth=0.75,
            capsize=1.8,
            zorder=6,
        )
        ax_a.annotate(
            row["gradient_id"],
            (
                row["mean_ring_count"] / x_scale,
                row["mean_reference_concentration_particles_ml"] / y_scale,
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6,
            color="#303A3F",
            zorder=7,
        )

    equation = (
        r"$C={:.2f}N{:+.2f}$".format(
            fit["slope"] / y_scale, fit["intercept"] / y_scale
        )
        + "\n"
        + r"$R^2={:.3f}$; LOOCV RMSE$={:.1f}\times10^3$".format(
            fit["r2"], metrics["loocv_rmse"] / y_scale
        )
        + "\n"
        + r"$n=18$ independent preparations"
    )
    ax_a.text(
        0.035,
        0.965,
        equation,
        transform=ax_a.transAxes,
        va="top",
        ha="left",
        fontsize=6.6,
        color="#1F2A30",
        linespacing=1.35,
    )
    ax_a.set_xlabel("Mean diffraction-ring count per field")
    ax_a.set_ylabel(r"Reference concentration ($\times10^3$ particles mL$^{-1}$)")
    ax_a.set_xlim(left=-1)
    ax_a.set_ylim(bottom=-8)
    ax_a.grid(axis="both", color="#D8DEE2", linewidth=0.45, alpha=0.7)
    handles, labels = ax_a.get_legend_handles_labels()
    keep = [0, 1, 2]
    ax_a.legend(
        [handles[i] for i in keep],
        [labels[i] for i in keep],
        loc="lower right",
        borderaxespad=0.4,
        handlelength=2.2,
    )

    max_value = max(
        prep["mean_reference_concentration_particles_ml"].max(),
        prep["loocv_predicted_concentration_particles_ml"].max(),
    )
    lim = (-5, math.ceil(max_value / 10_000) * 10 + 5)
    ax_b.plot(lim, lim, color="#7A858B", linestyle="--", linewidth=0.9)
    for gradient_id, group in prep.groupby("gradient_id", sort=False):
        ax_b.scatter(
            group["mean_reference_concentration_particles_ml"] / y_scale,
            group["loocv_predicted_concentration_particles_ml"] / y_scale,
            s=24,
            facecolor=palette[gradient_id],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.95,
        )
    ax_b.set_xlim(lim)
    ax_b.set_ylim(lim)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel(r"Reference ($\times10^3$ particles mL$^{-1}$)")
    ax_b.set_ylabel(r"LOOCV prediction ($\times10^3$ particles mL$^{-1}$)")
    ax_b.grid(axis="both", color="#D8DEE2", linewidth=0.45, alpha=0.7)
    ax_b.text(
        0.04,
        0.94,
        "MAE = {:.1f}".format(metrics["loocv_mae"] / y_scale),
        transform=ax_b.transAxes,
        va="top",
        fontsize=6.2,
        color="#303A3F",
    )

    for gradient_id, group in prep.groupby("gradient_id", sort=False):
        ax_c.scatter(
            group["fitted_concentration_particles_ml"] / y_scale,
            group["residual_particles_ml"] / y_scale,
            s=24,
            facecolor=palette[gradient_id],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.95,
        )
    ax_c.axhline(0, color="#7A858B", linestyle="--", linewidth=0.9)
    ax_c.set_xlabel(r"Fitted concentration ($\times10^3$ particles mL$^{-1}$)")
    ax_c.set_ylabel(r"Residual ($\times10^3$ particles mL$^{-1}$)")
    ax_c.grid(axis="both", color="#D8DEE2", linewidth=0.45, alpha=0.7)

    for label, ax in zip(("a", "b", "c"), (ax_a, ax_b, ax_c)):
        ax.text(
            -0.13,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
            ha="left",
        )

    base = OUTPUT_DIR / "quantitative_calibration_measured_data"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global OUTPUT_DIR, BOOTSTRAP_ITERATIONS, SEED
    parser = argparse.ArgumentParser(description="Measured PVC calibration; five fields per preparation, no imputation.")
    parser.add_argument("--input", type=Path, required=True, help="CSV or archived JSON with a 'values' table")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    OUTPUT_DIR, BOOTSTRAP_ITERATIONS, SEED = args.output_dir, args.bootstrap_iterations, args.seed
    if BOOTSTRAP_ITERATIONS < 1:
        parser.error("--bootstrap-iterations must be positive")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.input.suffix.lower() == ".csv":
        raw = pd.read_csv(args.input)
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        rows = payload["values"]
        raw = pd.DataFrame(rows[1:], columns=rows[0])
    if "field_id" in raw and raw.duplicated(["gradient_id", "preparation_id", "field_id"]).any():
        raise ValueError("Duplicate field rows would overcount technical replicates")

    numeric_columns = [
        "target_concentration_particles_ml",
        "hemocytometer_count_9_subsquares",
        "reference_concentration_particles_ml",
        "ring_count",
        "include_flag",
    ]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    summary_rows: list[dict[str, object]] = []
    for (gradient_id, preparation_id), group in raw.groupby(
        ["gradient_id", "preparation_id"], sort=False
    ):
        included = group[group["include_flag"].fillna(0).eq(1)]
        reference = included["reference_concentration_particles_ml"].dropna()
        rings = included["ring_count"].dropna()
        complete = len(included) == 5 and len(reference) == 5 and len(rings) == 5
        summary_rows.append(
            {
                "gradient_id": gradient_id,
                "preparation_id": preparation_id,
                "target_concentration_particles_ml": float(
                    group["target_concentration_particles_ml"].dropna().iloc[0]
                ),
                "n_reference_fields": len(reference),
                "n_ring_fields": len(rings),
                "mean_reference_concentration_particles_ml": (
                    float(reference.mean()) if len(reference) else np.nan
                ),
                "sd_reference_concentration_particles_ml": (
                    float(reference.std(ddof=1)) if len(reference) > 1 else np.nan
                ),
                "mean_ring_count": float(rings.mean()) if len(rings) else np.nan,
                "sd_ring_count": (
                    float(rings.std(ddof=1)) if len(rings) > 1 else np.nan
                ),
                "paired_complete": complete,
                "analysis_status": (
                    "Included"
                    if complete
                    else (
                        "Excluded: ring counts missing"
                        if len(reference) == 5 and len(rings) == 0
                        else "Excluded: paired measurements incomplete"
                    )
                ),
            }
        )

    all_preparations = pd.DataFrame(summary_rows)
    prep = all_preparations[all_preparations["paired_complete"]].copy()
    prep.reset_index(drop=True, inplace=True)
    prep.insert(0, "source_gradient_id", prep["gradient_id"])
    prep["gradient_id"] = prep["source_gradient_id"].map(FINAL_GRADIENT_MAP)
    if len(prep) < 4 or prep["gradient_id"].isna().any():
        raise ValueError("Need >=4 complete preparations with study gradient IDs C0,C1,C3,C5,C7,C8")

    x = prep["mean_ring_count"].to_numpy(dtype=float)
    y = prep["mean_reference_concentration_particles_ml"].to_numpy(dtype=float)
    fit = ols_fit(x, y)
    loocv_pred = leave_one_out_predictions(x, y)
    loocv_residual = y - loocv_pred
    loocv_sse = float(np.sum(loocv_residual**2))
    loocv_r2 = 1.0 - loocv_sse / float(np.sum((y - y.mean()) ** 2))

    slopes, intercepts = stratified_bootstrap(
        prep, BOOTSTRAP_ITERATIONS, SEED
    )
    bootstrap_slope_ci = np.percentile(slopes, [2.5, 97.5])
    bootstrap_intercept_ci = np.percentile(intercepts, [2.5, 97.5])
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    prep["fitted_concentration_particles_ml"] = fit["fitted"]
    prep["residual_particles_ml"] = fit["residuals"]
    prep["loocv_predicted_concentration_particles_ml"] = loocv_pred
    prep["loocv_residual_particles_ml"] = loocv_residual

    gradient = (
        prep.groupby("gradient_id", sort=False)
        .agg(
            target_concentration_particles_ml=(
                "target_concentration_particles_ml",
                "mean",
            ),
            n_preparations=("preparation_id", "count"),
            mean_reference_concentration_particles_ml=(
                "mean_reference_concentration_particles_ml",
                "mean",
            ),
            sd_reference_concentration_particles_ml=(
                "mean_reference_concentration_particles_ml",
                "std",
            ),
            mean_ring_count=("mean_ring_count", "mean"),
            sd_ring_count=("mean_ring_count", "std"),
        )
        .reset_index()
    )

    x_grid = np.linspace(max(0, x.min() - 1), x.max() + 2, 240)
    y_grid = fit["intercept"] + fit["slope"] * x_grid
    mean_se = fit["residual_se"] * np.sqrt(
        1.0 / fit["n"] + (x_grid - fit["x_bar"]) ** 2 / fit["sxx"]
    )
    pred_se = fit["residual_se"] * np.sqrt(
        1.0 + 1.0 / fit["n"] + (x_grid - fit["x_bar"]) ** 2 / fit["sxx"]
    )
    curve = pd.DataFrame(
        {
            "mean_ring_count": x_grid,
            "fitted_concentration_particles_ml": y_grid,
            "confidence_low": y_grid - fit["t_crit"] * mean_se,
            "confidence_high": y_grid + fit["t_crit"] * mean_se,
            "prediction_low": y_grid - fit["t_crit"] * pred_se,
            "prediction_high": y_grid + fit["t_crit"] * pred_se,
        }
    )

    origin_slope = float(np.sum(x * y) / np.sum(x * x))
    origin_predictions = origin_slope * x
    origin_residuals = y - origin_predictions
    origin_rmse = math.sqrt(float(np.mean(origin_residuals**2)))
    origin_mae = float(np.mean(np.abs(origin_residuals)))
    origin_r2_centered = 1.0 - float(np.sum(origin_residuals**2)) / fit["sst"]

    metrics = {
        "n_gradients_included": int(prep["gradient_id"].nunique()),
        "n_preparations_included": int(len(prep)),
        "n_fields_per_preparation": 5,
        "slope_particles_ml_per_ring": float(fit["slope"]),
        "intercept_particles_ml": float(fit["intercept"]),
        "r2": float(fit["r2"]),
        "adjusted_r2": float(fit["adjusted_r2"]),
        "rmse_particles_ml": float(fit["rmse"]),
        "mae_particles_ml": float(fit["mae"]),
        "residual_standard_error_particles_ml": float(fit["residual_se"]),
        "slope_analytical_ci95_low": float(fit["slope_ci_low"]),
        "slope_analytical_ci95_high": float(fit["slope_ci_high"]),
        "intercept_analytical_ci95_low": float(fit["intercept_ci_low"]),
        "intercept_analytical_ci95_high": float(fit["intercept_ci_high"]),
        "slope_stratified_bootstrap_ci95_low": float(bootstrap_slope_ci[0]),
        "slope_stratified_bootstrap_ci95_high": float(bootstrap_slope_ci[1]),
        "intercept_stratified_bootstrap_ci95_low": float(
            bootstrap_intercept_ci[0]
        ),
        "intercept_stratified_bootstrap_ci95_high": float(
            bootstrap_intercept_ci[1]
        ),
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "loocv_rmse": math.sqrt(float(np.mean(loocv_residual**2))),
        "loocv_mae": float(np.mean(np.abs(loocv_residual))),
        "loocv_r2": loocv_r2,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "through_origin_slope_particles_ml_per_ring": origin_slope,
        "through_origin_rmse_particles_ml": origin_rmse,
        "through_origin_mae_particles_ml": origin_mae,
        "through_origin_centered_r2": origin_r2_centered,
    }

    metric_frame = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )
    excluded = all_preparations[~all_preparations["paired_complete"]].copy()

    all_preparations.to_csv(
        OUTPUT_DIR / "quantitative_all_preparation_status.csv",
        index=False,
        encoding="utf-8-sig",
    )
    prep.to_csv(
        OUTPUT_DIR / "quantitative_preparation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gradient.to_csv(
        OUTPUT_DIR / "quantitative_gradient_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metric_frame.to_csv(
        OUTPUT_DIR / "quantitative_regression_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    curve.to_csv(
        OUTPUT_DIR / "quantitative_regression_curve.csv",
        index=False,
        encoding="utf-8-sig",
    )
    excluded.to_csv(
        OUTPUT_DIR / "quantitative_excluded_measurements.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analysis_payload = {
        "metrics": metrics,
        "preparations": json.loads(prep.to_json(orient="records")),
        "gradients": json.loads(gradient.to_json(orient="records")),
        "all_preparations": json.loads(
            all_preparations.to_json(orient="records")
        ),
    }
    (OUTPUT_DIR / "quantitative_analysis_results.json").write_text(
        json.dumps(analysis_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    legend = f"""Figure | Quantitative calibration of diffraction-ring counts against hemocytometer-derived concentration.
(a) Mean diffraction-ring count per field plotted against the reference microsphere concentration for the six final measured concentration gradients (G0-G5). Each circle represents one independent slide preparation and is the mean of five non-overlapping fields; diamonds show the mean across three preparations, with horizontal and vertical error bars denoting s.d. The solid line is the ordinary least-squares fit, the darker band is the 95% confidence interval of the mean response, and the lighter band is the 95% prediction interval. No concentration values were interpolated or imputed. (b) Leave-one-preparation-out cross-validated predictions versus reference concentration; the dashed line indicates identity. (c) Residuals from the fitted calibration model. n = {len(prep)} independent preparations, with five technical fields per preparation. Model: C = {fit['slope']:.2f}N {fit['intercept']:+.2f} particles mL-1, where N is mean ring count per field; R2 = {fit['r2']:.3f}; LOOCV RMSE = {metrics['loocv_rmse']:.1f} particles mL-1."""
    (OUTPUT_DIR / "quantitative_calibration_figure_legend.txt").write_text(
        legend, encoding="utf-8"
    )

    qa_notes = f"""Core conclusion: Across the six final measured gradients, five-field mean diffraction-ring count is strongly associated with hemocytometer-derived concentration.
Archetype: quantitative grid with a dominant calibration panel and two validation panels.
Statistical unit: independent preparation (n={len(prep)}); five fields are technical replicates averaged within each preparation.
Final gradients: {', '.join(prep['gradient_id'].drop_duplicates())}; source identifiers retained in the source_gradient_id column for traceability.
Model direction: reference concentration as the outcome and mean ring count as the predictor.
Primary interval: analytical 95% confidence and prediction intervals from OLS.
Robustness: leave-one-preparation-out cross-validation and {BOOTSTRAP_ITERATIONS:,}-iteration gradient-stratified bootstrap.
Limitation: analytical sensitivity, LOD and LOQ are not claimed because the blank variance is zero in the current observations and independent low-concentration repeats are insufficient.
Exports: SVG and PDF with editable text, 600-dpi TIFF, and 300-dpi PNG preview.
"""
    (OUTPUT_DIR / "quantitative_calibration_qa_notes.txt").write_text(
        qa_notes, encoding="utf-8"
    )

    save_publication_figure(prep, gradient, fit, curve, metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
