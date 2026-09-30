# Edge Deployment and Acquisition

Do not install desktop GPU wheels on ARM devices. Match JetPack/PyTorch/TensorRT on Jetson; match RKNN Toolkit2, Lite2 and the board's RKNPU driver on RK3588. Python, NumPy, OpenCV, Pillow and pyserial are required for acquisition/inference. Keep the repository layout when copying source to a device. No IP addresses, login credentials, binary runtimes or automatic remote configuration are included.

## Saved-Sequence Desktop Check

With trusted trained assets placed as described in `models/README.md`:

```bash
python jetson_hsi_end_to_end.py --input-dir data/example_sequence --output-dir results/example --detector-weights models/yolo13n_pose_lscd_lqe_best.pt --classifier-numpy models/deployment_classifier_domain_adapted.npz --selected-wavelengths experiment_configs/selected_wavelengths_real_train_only.csv --imgsz 768 1024 --device cpu --no-save-cubes
```

This reference path writes a final annotated image, per-object labels/counts, and a JSON summary. Optional windows require an actual GUI session; use `--show-windows` only then. Processing a saved sequence does not move hardware. The selected wavelength CSV must match the classifier artifact exactly.

## RK3588 Conversion

On an export workstation, provide PyTorch and ONNX. Run RKNN compilation in a separate supported Toolkit2 environment, not in the board's Lite2 environment:

```bash
python deployment/rk3588/conversion/export_raw_pose_onnx.py --weights models/yolo13n_pose_lscd_lqe_best.pt --ultralytics-source vendor/ultralytics --output results/export/pose_raw_stack.onnx --height 768 --width 1024 --without-lqe --approximate-gelu --upsample-stack-scales
python deployment/rk3588/conversion/export_lqe_weights.py --weights models/yolo13n_pose_lscd_lqe_best.pt --ultralytics-source vendor/ultralytics --output deployment/rk3588/models/yolo13n_pose_lqe_weights.npz
python deployment/rk3588/conversion/convert_onnx_to_rk3588.py --onnx results/export/pose_raw_stack.onnx --output deployment/rk3588/models/yolo13n_pose_lscd_stack_tanhgelu_768x1024_rk3588_fp16.rknn
```

Here `--without-lqe` removes LQE from the **NPU graph**, not the complete algorithm. The host loads the exported LQE MLP, then performs quality correction, DFL/pose decode and NMS in NumPy. GELU uses the tanh approximation for the target conversion. Default conversion is FP16, not INT8. Revalidate conversion accuracy and center matching after changing toolkit, shape, approximation or quantization. The raw stack is `[1,204,96,128]` for 768 x 1024 input.

Put the compatible classifier and LQE `.npz` files in `deployment/rk3588/models/`. The wavelength config is supplied in `deployment/rk3588/config/`. On the RK device:

```bash
cd deployment/rk3588
bash run_rk3588_hsi_once.sh /path/to/sequence /path/to/results --no-save-cubes --minimal-output
```

`PYTHON_BIN` may select the Lite2-enabled interpreter. Avoid `--show-windows` during timing: a zero popup delay waits for human input. The minimal-output setting skips redundant plots/contact sheets and full-cube export; changing these flags changes the timed work. Inspect `pipeline_summary.json` to see stage boundaries.

## Jetson TensorRT

Export the standard decoded detector graph on the workstation:

```bash
python export_jetson_onnx.py --weights models/yolo13n_pose_lscd_lqe_best.pt --ultralytics-source vendor/ultralytics --output-dir deployment/jetson/artifacts --height 768 --width 1024 --device cpu
```

Copy the repository and assets to the Jetson. Build the engine **on the target device** with its installed TensorRT:

```bash
cd deployment/jetson
bash build_tensorrt_engine.sh artifacts/yolo13n_pose_lscd_lqe_768x1024_fp32.onnx artifacts
bash run_jetson_hsi_once.sh /path/to/sequence /path/to/results --no-save-cubes
```

The build script wraps a raw TensorRT engine with the metadata expected by the retained Ultralytics loader. Do not mix raw/wrapped engines or assume an engine is portable across hardware/runtime versions. Put `deployment_classifier_domain_adapted.npz` in this platform's `models/` folder. A compatible Ultralytics/PyTorch runtime is still needed by the reference engine loader.

This is the reference Jetson workflow. Its inclusion does not establish a new equal-work benchmark against the optimized RK pipeline, nor reproduce a particular manuscript timing value automatically. Match outputs, preprocessing, threads, cache state, warm-up, and timing boundaries before comparing devices.

## Camera and Translation Stage

Start with a saved sequence, then camera preview, then a carefully supervised scan. The OpenMV board must run compatible image-server firmware (normally saved as `main.py` for power-on startup). That board firmware is not supplied here. Host protocol: request byte `s`; response header `AA55`; 32-bit little-endian JPEG length; chunked JPEG with `k` acknowledgements. The CMOS is accessed through the OpenMV serial interface, not an arbitrary USB webcam API.

Linux example from either platform's deployment folder:

```bash
bash tools/list_serial_devices.sh
python3 acquisition/acquire_and_infer.py --camera-port /dev/ttyACM0 --preview-only
python3 acquisition/acquire_and_infer.py --help
```

Use stable `/dev/serial/by-id/` names where available. `/dev/ttyACM0` and `/dev/ttyUSB0` are examples, not guaranteed assignments. Verify serial permissions and CH341 support in the board kernel. Do not assume that seeing a USB adapter in `lsusb` means a serial driver is active.

**Motion safety:** confirm controller protocol, travel direction, origin and hard limits first. The retained stage commands are for the study controller and can move it immediately after confirmation. Leave confirmation enabled and never use unattended `--assume-yes` while bringing up new hardware.

For the original RK board's Weston desktop, the retained preview/scan shell wrappers set board-specific Wayland/fullscreen defaults. On another Linux desktop use the Python entry point in the logged-in GUI session and its `--help` options instead of forcing the root Weston socket. Once the camera preview and stage checks succeed:

```bash
# Original RK board with the verified controller and both ports:
bash run_rk3588_hardware_scan.sh /dev/ttyACM0 /dev/ttyUSB0
# Jetson alternative:
bash run_jetson_hardware_scan.sh /dev/ttyACM0 /dev/ttyUSB0
```

The nominal acquisition initializes at -900000 and records 400 frames after successive 4000-unit moves, ending at 700000. Position units and wavelength calibration belong to this instrument. The application validates a complete sequence before analysis. Test protocol code does not access serial devices or run motion.

## Concentration and Performance Reporting

Defaults preserve the measured six-gradient PVC calibration coefficients. Refit for a new loading/optical setup and supply `--concentration-slope` and `--concentration-intercept`; the geometric alternative requires measured effective field/wet area and volume. Values for other particles are not validated absolute concentrations merely because classification succeeds.

Report inference-only latency separately from full saved-sequence analysis. Acquisition duration is a third quantity. Use multiple timed runs after warm-up, sample standard deviation, identical input/outputs, and record software/driver/model hashes. Missing calibrated power sensing is not a license to infer watts from temperature or nominal chip specifications.
