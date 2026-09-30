# Model Artifacts (Not Bundled)

Expected files for the inference examples:

| File | Meaning |
| --- | --- |
| `yolo13n_pose_lscd_lqe_best.pt` | Trusted trained DiffCenter-YOLO Pose checkpoint |
| `deployment_classifier_domain_adapted.npz` | Linear-SVM parameters, feature scaling and class order |
| `yolov13n.pt` | YOLO13 COCO detection pretraining checkpoint, not pose pretraining |

RK3588 additionally needs `yolo13n_pose_lqe_weights.npz` and `yolo13n_pose_lscd_stack_tanhgelu_768x1024_rk3588_fp16.rknn` in `deployment/rk3588/models/`. Jetson needs a locally built TensorRT engine in `deployment/jetson/artifacts/`. Each device uses its own `models/deployment_classifier_domain_adapted.npz` copy and `config/selected_wavelengths_real_train_only.csv`.

Generative-model training produces `cube_autoencoder.pt` and `latent_ddpm.pt` beneath each run's `checkpoints/` folder. Keep the accompanying normalization, wavelengths, class order and configuration. Do not reuse generative checkpoints across corrected/masked and unmasked-square domains without a new validation.

This repository distributes code only. Supply your own trusted weights and model artifacts; `.gitignore` excludes them from source commits.
