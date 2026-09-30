# Packaging Verification

Checked on 30 September 2026. These checks validate the source preparation and selected numerical implementations; they do not constitute a rerun of all manuscript experiments.

## Completed Checks

| Check | Observed result |
| --- | --- |
| Unit tests | 33 passed; includes ENVI round-trip, invalid split rejection, training/test discovery, calibration, paired statistics, assignment, fixed-center conversion, Pose forward/backward and serial framing |
| Documented CLI entry points | 22 `--help` checks passed without accessing hardware |
| Artificial PI-LDM training and export | One autoencoder epoch, one diffusion epoch, four diffusion steps; five finite ENVI cubes exported |
| Trained detector checkpoint compatibility | Strict state-dictionary loading into the public YOLO13n-Pose-LSCD-LQE YAML succeeded; maximum output difference 0.0 on a fixed 128 x 128 test input |
| YOLO13 detection pretraining initialization | 778 of 854 state entries transferred to the initial Pose-LSCD-LQE model |
| Official baseline snapshot | v8 and v26 Pose forward passes succeeded in a separate process; snapshot version 8.4.101 |
| Custom baseline snapshot | v11, v12 and v13 Pose plus v13 LSCD/LQE combinations passed CPU forward tests; snapshot version 8.3.9 |
| Raw stacked ONNX export | ONNX checker passed; ONNX Runtime output agreed with PyTorch on a fixed random input at rtol/atol 0.0001 |
| Archived measured calibration | 18 complete preparations; slope 7177.958896475087, intercept 16.595417284901487, R-squared 0.9911542145285686, leave-one-preparation-out R-squared 0.9893636536506806 |

The ONNX trace emits two shape-related tracer warnings. The checked graph is intentionally fixed-shape; the check does not establish a dynamic-shape export. Device compilation and numerical parity must still be checked for the target deployment shape and runtime.

## Observed Local Environment

CPU checks ran on Windows with Python 3.12.7. The accompanying CI definition targets Python 3.10 but has not yet run on GitHub.

| Package | Version |
| --- | --- |
| PyTorch / torchvision | 2.10.0 / 0.25.0 |
| NumPy / SciPy | 1.26.4 / 1.13.1 |
| pandas / scikit-learn | 2.2.2 / 1.7.1 |
| matplotlib / Pillow | 3.9.2 / 10.4.0 |
| opencv-python | 4.11.0.86 |
| ONNX / ONNX Runtime | 1.20.1 / 1.24.3 |
| pytest / dill / ultralytics-thop | 7.4.4 / 0.3.8 / 2.0.18 |

Existing local scientific artifacts were used only for checkpoint and calibration consistency checks. They are not included in the source ZIP. A fresh dependency installation, full-duration training, a fresh independent biological test, RKNN compilation, TensorRT compilation, real serial acquisition, and new device timing/resource measurements have not been performed during packaging.

The source audit parses included Python files, excludes datasets/caches/binary model files, and scans application sources for the specified private-address/path/key patterns.
