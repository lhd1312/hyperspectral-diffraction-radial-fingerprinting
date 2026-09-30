# Hyperspectral Diffraction Radial Fingerprinting

Research code accompanying **Hyperspectral Diffraction Radial Fingerprinting for Label-Free Identification and Quantification of Crop Pathogen Spores**.

Repository: https://github.com/lhd1312/hyperspectral-diffraction-radial-fingerprinting

[Reproduction](docs/REPRODUCIBILITY.md) | [Data format](docs/DATA_FORMAT.md) | [Edge deployment](docs/DEPLOYMENT.md)

## Scope

The workflow combines hyperspectral diffraction measurements, physics-informed latent diffusion augmentation, particle-center detection, spectral-radial classification, and count-to-concentration calibration.

```text
Training ROIs -> conditional autoencoder -> latent DDPM / physics-informed LDM
             -> training augmentation -> wavelength / classifier comparisons

Raw wavelength sequence -> pseudo-RGB -> DiffCenter-YOLO centers
                        -> fixed 100 x 100 ROIs across wavelengths
                        -> spectral-radial features -> class labels
                        -> counts -> calibrated concentration estimate
```

| Component | Main entry point |
| --- | --- |
| PI-LDM and unconstrained LDM | `pinn_diffusion_hsi_augmentation.py` |
| Real-only / classical / LDM / PI-LDM comparison, bootstrap and McNemar | `evaluate_augmentation_classification.py` |
| Spectral/spatial ablations and CNN comparisons | `run_unified_hsi_classification.py` |
| Seven wavelength selection methods | `run_feature_wavelength_selection_comparison.py` |
| Fixed-center annotation and Pose conversion | `prepare_yolo_hsi_annotation.py`, `convert_fixed_boxes_to_pose.py` |
| Pose baselines and LSCD/LQE ablations | `train_pose_architecture_benchmark.py` |
| Class-independent deployment classifier | `train_class_independent_deployment_classifier.py` |
| Reviewed full-field adaptation/evaluation | `run_fullfield_predeployment_validation.py` |
| Measured PVC calibration and leave-one-preparation-out evaluation | `analyze_measured_calibration.py` |
| RK3588 / Jetson inference and hardware acquisition | `deployment/` |

The legacy filename `PINN-DDPM` is retained for traceability. The implementation is a **physics-informed latent diffusion model (PI-LDM)** with Fresnel-inspired radial scaling and additional priors; it does **not** solve a full complex-wave Fresnel PDE residual. See [limitations](docs/LIMITATIONS.md).

## Installation

Use a separate Python 3.10 or 3.11 environment. Install a matching PyTorch/torchvision build for your platform, then the application dependencies. Example for a CPU-only code check:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pytest -q
python tools/smoke_augmentation.py
```

The smoke command creates explicitly artificial test cubes and runs one tiny autoencoder/diffusion training cycle, including ENVI export. It is **not experimental evidence** and does not reproduce manuscript accuracy. No camera, translation stage, private data, or pretrained weights are required for these checks.

Detector commands select the bundled framework source themselves. **Do not replace it with `pip install ultralytics`**: the custom YOLO13/LSCD/LQE class definitions and checkpoint namespaces are needed. The selected custom fork and the newer official baseline snapshot are separated in `vendor/`; run each model in a new Python process. These are scoped source snapshots, not general-purpose replacements for upstream packages.

## Reproducing the Study

1. Prepare the measured data and fixed train/test manifests described in [Data format](docs/DATA_FORMAT.md).
2. Follow [Reproduction](docs/REPRODUCIBILITY.md), starting with augmentation of **training ROIs only**.
3. Use [Deployment](docs/DEPLOYMENT.md) for conversion, saved-sequence inference, and optional acquisition.
4. Read [Limitations](docs/LIMITATIONS.md) before interpreting generalization, calibration, or timing.

This repository contains source code and configuration. Experimental data, labels, trained checkpoints and device-specific engines are excluded. [Model assets](models/README.md) describes the files needed for inference.

## Citation and Licensing

Please cite the manuscript title above and the exact repository release/commit used. Publication metadata and an archival DOI will be added when available; no author list is included in this draft. See [citation](docs/CITATION.md).

Bundled Ultralytics sources carry AGPL-3.0 notices. The accompanying [LICENSE](LICENSE) preserves the upstream license text. Implementation sources are described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Only load `.pt` and pickle artifacts from a trusted source. Hardware scan commands can move a physical stage; inspect limits and controller compatibility before use.
