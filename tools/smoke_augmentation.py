"""Run a tiny, artificial PI-LDM training/export check; not experimental data."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_diffusion_hsi_augmentation import CATEGORY_SPECS, EnviMeta, read_envi_cube, write_envi_cube


def run(work: Path) -> None:
    rng = np.random.default_rng(2026)
    meta = EnviMeta(32, 32, 8, 4, "bsq", 0, np.linspace(411, 779, 8))
    yy, xx = np.mgrid[-1:1:32j, -1:1:32j]
    radius = np.hypot(xx, yy)
    for index, (_, folder, label) in enumerate(CATEGORY_SPECS):
        for sample in range(3):
            cube = np.stack([.5 + .2 * np.cos(radius * (10 + index + band * .1)) for band in range(8)])
            cube += rng.normal(0, .02, cube.shape)
            split = folder + ("_test" if sample == 2 else "")
            write_envi_cube(work / "data" / split / f"{label}_{sample}.hdr", cube, meta,
                            "ARTIFICIAL SOFTWARE TEST ONLY - NOT A MEASUREMENT")
    subprocess.run([sys.executable, str(ROOT / "pinn_diffusion_hsi_augmentation.py"),
                    "--data-root", str(work / "data"), "--run-root", str(work / "run"),
                    "--output-root", str(work / "generated"), "--device", "cpu", "--torch-threads", "2",
                    "--epochs-autoencoder", "1", "--epochs-diffusion", "1", "--timesteps", "4",
                    "--latent-dim", "8", "--base-channels", "8", "--radial-bins", "8",
                    "--per-class", "1", "--batch-size", "2", "--latent-batch-size", "5",
                    "--values-per-cube", "512", "--diffusion-physics-every", "1"], check=True, cwd=ROOT)
    outputs = list((work / "generated").rglob("*.hdr"))
    assert len(outputs) == 5, f"Expected one generated cube per class, got {len(outputs)}"
    for path in outputs:
        _, cube = read_envi_cube(path)
        assert cube.shape == (8, 32, 32) and np.isfinite(cube).all()
    print("PASS: artificial training, physics loss, latent sampling and five ENVI exports")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="Retain outputs in a new .smoke subdirectory")
    args = parser.parse_args()
    parent = ROOT / ".smoke"
    parent.mkdir(exist_ok=True)
    if args.keep:
        run(Path(tempfile.mkdtemp(prefix="artificial_", dir=parent)))
    else:
        with tempfile.TemporaryDirectory(prefix="artificial_", dir=parent) as directory:
            run(Path(directory))


if __name__ == "__main__":
    main()
