"""Check documented CLI entry points without training, inference or hardware access."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "pinn_diffusion_hsi_augmentation.py", "evaluate_augmentation_classification.py",
    "compare_classifiers_for_deployment.py", "run_unified_hsi_classification.py",
    "run_feature_wavelength_selection_comparison.py", "train_class_independent_deployment_classifier.py",
    "run_fullfield_predeployment_validation.py", "prepare_yolo_hsi_annotation.py",
    "review_yolo26_label_review.py", "convert_fixed_boxes_to_pose.py", "jetson_hsi_end_to_end.py",
    "export_jetson_onnx.py", "export_pose_center_predictions.py", "evaluate_pose_threshold_sweep.py",
    "train_pose_architecture_benchmark.py", "analyze_measured_calibration.py",
    "deployment/rk3588/pipeline/rk3588_hsi_end_to_end.py",
    "deployment/rk3588/conversion/export_raw_pose_onnx.py",
    "deployment/rk3588/conversion/export_lqe_weights.py",
    "deployment/rk3588/conversion/convert_onnx_to_rk3588.py",
    "deployment/rk3588/acquisition/acquire_and_infer.py",
    "deployment/jetson/acquisition/acquire_and_infer.py",
]


def main():
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    failed = []
    for script in SCRIPTS:
        result = subprocess.run([sys.executable, str(ROOT / script), "--help"], cwd=ROOT,
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", timeout=90)
        print(f"{'PASS' if result.returncode == 0 else 'FAIL'} {script}", flush=True)
        if result.returncode:
            failed.append(script)
            print(result.stdout)
    if failed:
        raise SystemExit(f"{len(failed)} CLI checks failed")
    print(f"PASS: {len(SCRIPTS)} CLI help checks")


if __name__ == "__main__":
    main()
