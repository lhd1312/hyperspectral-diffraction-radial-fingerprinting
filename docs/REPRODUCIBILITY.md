# Reproduction Guide

Run commands from the repository root in a prepared environment. Commands below use one-line syntax that also works in PowerShell. Input paths are examples; no private measurements or trained weights are downloaded automatically. Use separate output folders for different seeds, data versions, or configurations.

## 1. Training-Only Latent Diffusion

```bash
python pinn_diffusion_hsi_augmentation.py --data-root ENVI_CORRECTED_CROPPED --fit-subset train --per-class 100 --epochs-autoencoder 80 --epochs-diffusion 300 --timesteps 200 --batch-size 2 --latent-batch-size 32 --seed 2026 --run-root pinn_diffusion_runs --output-root ENVI_PINN_DIFFUSION_AUGMENTED
python pinn_diffusion_hsi_augmentation.py --data-root ENVI_CORRECTED_CROPPED --fit-subset train --no-physics --per-class 100 --epochs-autoencoder 80 --epochs-diffusion 300 --timesteps 200 --batch-size 2 --latent-batch-size 32 --seed 2026 --run-root ddpm_diffusion_runs --output-root ENVI_DDPM_AUGMENTED
```

Common settings retained from the experiment: latent size 128, base channels 32, AdamW learning rates 0.0002, weight decay 0.0001, percentiles 0.5/99.5 for training-only intensity normalization, 48 radial bins, linear beta schedule 0.0001-0.02. The default AE epoch count is 60, so the command **explicitly sets the experimental 80 epochs**. PI-LDM uses AE physics weight 0.1 and diffusion physics weight 0.02 every fourth step; `--help` lists individual priors. Unconstrained LDM disables all physics weights and trains its own autoencoder and denoiser.

Do not reuse a PI-LDM checkpoint for the unconstrained baseline, change a dataset and reuse its cache, or fit on test folders. Saved run configurations and generated manifests should accompany numerical results. Historical optical geometry defaults are 60 cm source-to-pinhole, 60 cm pinhole-to-object, and 5 mm object-to-sensor. Later 1 mm quantification measurements are a separate geometry and must not be silently conflated.

## 2. Augmentation and Classifier Comparisons

```bash
python evaluate_augmentation_classification.py --data-root ENVI_CORRECTED_CROPPED --pinn-dir ENVI_PINN_DIFFUSION_AUGMENTED --ddpm-dir ENVI_DDPM_AUGMENTED --out-root results/augmentation --augmented-per-class 100 --classical-per-class 100 --primary-classifier svm_rbf --bootstrap-iterations 5000 --seed 2026
python run_unified_hsi_classification.py --data-root ENVI_CORRECTED_CROPPED --pinn-root ENVI_PINN_DIFFUSION_AUGMENTED --output-root predeployment_validation --selected-bands 40
python run_feature_wavelength_selection_comparison.py --validation-root predeployment_validation --band-counts 5 10 20 30 40
python compare_classifiers_for_deployment.py --data-root ENVI_CORRECTED_CROPPED --pinn-dir ENVI_PINN_DIFFUSION_AUGMENTED --out-root results/classifiers
```

The first command compares real-only, classical augmentation, unconstrained LDM, and PI-LDM on real held-out ROIs, and includes paired bootstrap/McNemar analyses. The unified command builds feature caches and compares spectral/spatial representations and 1D/2D/3D CNNs. The wavelength command consumes those caches and evaluates ANOVA, MI, RF, PLS-VIP, SPA, ReliefF, and 2D-COS rankings. Selection scores are fitted on real training data. These runs may be computationally expensive; there is no promise of bitwise-identical cross-platform training.

The CNN experiment uses a real-training internal validation split while synthetic training data are generated from the full real-training pool. The internal validation is therefore not wholly independent of generator fitting. The held-out real test set remains separate, but do not characterize the internal procedure as fully nested validation. Selecting the best architecture or feature count after inspecting test results requires a fresh external test for an unbiased final selection claim.

## 3. Diffraction-Center Detector

```bash
python prepare_yolo_hsi_annotation.py --oridata-root oridata --out-root data/annotation --label-mode particle --box-size 100
python convert_fixed_boxes_to_pose.py --review-root data/annotation --source-label-dir labels --output-root data/pose --copy-images
python tools/make_pose_yaml.py --root data/pose --output data/pose/data.yaml
python train_pose_architecture_benchmark.py yolo13n-pose-lscd-lqe --data data/pose/data.yaml --pretrained models/yolov13n.pt --seed 2027 --name yolo13n-pose-lscd-lqe-seed2027
```

Review fixed centers and missed objects before conversion; automatic candidates are not ground truth. Copying images avoids symlink restrictions on Windows. The conversion tool expects the review dataset layout; inspect `--help` before converting a different annotation export.

The training script supports `yolov8n-pose`, `yolo11n-pose`, `yolo12n-pose`, `yolo13n-pose`, `yolo26n-pose`, and YOLO13 `-lscd`, `-lqe`, `-lscd-lqe`. Run each in a separate process with its own checkpoint and output folder. YOLO13 initialization uses **COCO detection** weights `yolov13n.pt`, not an invented pose-pretrained checkpoint. YOLO12 also uses detection pretraining; v8/11/26 use their pose checkpoints. `--scratch` is for code tests, not the paper's transfer-learning comparison.

Retained settings: image size 1024, batch 4, epochs 150, patience 40, optimizer `auto`, deterministic mode, AMP off, HSV h/s=0 and v=0.15, translation 0.05, scale 0.1, vertical/horizontal flips 0.5, mosaic 0.2, mosaic closed for the final 10 epochs, mixup/copy-paste disabled. Final validation uses confidence 0.001, IoU 0.7, maximum 500 detections, FP32, plots disabled. Record the actual optimizer chosen by `auto`, seed, framework, and weights hash. Compare repeated seeds rather than only a favorable run.

`best_fp32.pt` is used when present, otherwise `best.pt`. Report **box metrics and pose metrics separately**. Pose OKS mAP and F1 within a 10 px center radius are not interchangeable.

## 4. Class-Independent Full-Field Pipeline

```bash
python train_class_independent_deployment_classifier.py --data-root data/uncorrected_rois --train-per-class 70 --test-per-class 30 --top-wavelengths 40 --radial-bins 12
python run_fullfield_predeployment_validation.py --help
python jetson_hsi_end_to_end.py --help
```

The deployment training command consumes the **uncorrected** numeric-suffix ROI layout, not the corrected augmentation tree. Full-field adaptation additionally requires the original registered wavelength sequences, reviewed detection labels, split manifest, detector checkpoint, and initial deployment classifier. Supply those explicitly using its CLI; the script's historical relative default paths describe the original workflow, not assets included here. Only reviewed training-field crops may enter adaptation; independent fields remain held out.

The deployed classifier is a NumPy-exported **linear SVM** with class-independent spectral-radial features. The RBF-SVM used in ROI comparison tables is a different estimator. Do not combine an RBF artifact with the linear decoder or replace its feature preprocessing.

## 5. Measured PVC Calibration

```bash
python analyze_measured_calibration.py --input data/measured_calibration.csv --output-dir results/calibration --bootstrap-iterations 10000 --seed 20260721
```

The script aggregates five fields per preparation, excludes incomplete preparations, fits an intercept-containing OLS line, reports confidence/prediction intervals and residuals, resamples preparations within gradients, and performs leave-one-preparation-out prediction. It exports source tables and figures. Enter measured values only; the template contains no surrogate observations. See [limitations](LIMITATIONS.md) for count and concentration interpretation.

## Audit Trail

`VERIFICATION.md` records the code checks performed during preparation. Record the repository commit together with each run's configuration, software environment, input split and model hashes when reproducing an experiment.
