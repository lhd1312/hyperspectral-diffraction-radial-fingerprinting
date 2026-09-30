import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent
CUSTOM_SOURCE = ROOT / "vendor" / "ultralytics"
OFFICIAL_SOURCE = ROOT / "vendor" / "ultralytics-official"
DATA_YAML = ROOT / "yolo_hsi_pose_center_dataset_v2/data.yaml"
TRAIN_ROOT = ROOT / "runs/pose_architecture_benchmark_local"
VAL_ROOT = ROOT / "runs/pose_architecture_benchmark_local_val"
EXPERIMENT_CONFIGS = ROOT / "experiment_configs"


MODEL_SPECS = {
    "yolov8n-pose": {
        "source": OFFICIAL_SOURCE,
        "yaml": OFFICIAL_SOURCE / "ultralytics/cfg/models/v8/yolov8n-pose.yaml",
        "yaml_source": OFFICIAL_SOURCE / "ultralytics/cfg/models/v8/yolov8-pose.yaml",
        "weights": ROOT / "yolov8n-pose.pt",
        "pretraining": "COCO pose",
    },
    "yolo11n-pose": {
        "source": CUSTOM_SOURCE,
        "yaml": CUSTOM_SOURCE / "ultralytics/cfg/models/11/yolo11n-pose.yaml",
        "yaml_source": CUSTOM_SOURCE / "ultralytics/cfg/models/11/yolo11-pose.yaml",
        "weights": ROOT / "yolo11n-pose.pt",
        "pretraining": "COCO pose",
    },
    "yolo12n-pose": {
        "source": CUSTOM_SOURCE,
        "yaml": CUSTOM_SOURCE / "ultralytics/cfg/models/12/yolo12n-pose.yaml",
        "yaml_source": CUSTOM_SOURCE / "ultralytics/cfg/models/12/yolo12-pose.yaml",
        "weights": ROOT / "yolo12n.pt",
        "pretraining": "COCO detection; pose head initialized from scratch",
    },
    "yolo13n-pose": {
        "source": CUSTOM_SOURCE,
        "yaml": CUSTOM_SOURCE / "ultralytics/cfg/models/13/yolo13n-pose.yaml",
        "yaml_source": CUSTOM_SOURCE / "ultralytics/cfg/models/13/yolo13-pose.yaml",
        "weights": CUSTOM_SOURCE / "yolov13n.pt",
        "pretraining": "COCO detection; pose head initialized from scratch",
    },
    "yolo13n-pose-lscd": {
        "source": CUSTOM_SOURCE,
        "yaml": EXPERIMENT_CONFIGS / "yolo13n-pose-LSCD.yaml",
        "yaml_source": EXPERIMENT_CONFIGS / "yolo13-pose-LSCD.yaml",
        "weights": CUSTOM_SOURCE / "yolov13n.pt",
        "pretraining": "COCO detection; LSCD pose head initialized from scratch",
    },
    "yolo13n-pose-lqe": {
        "source": CUSTOM_SOURCE,
        "yaml": EXPERIMENT_CONFIGS / "yolo13n-pose-LQE.yaml",
        "yaml_source": EXPERIMENT_CONFIGS / "yolo13-pose-LQE.yaml",
        "weights": CUSTOM_SOURCE / "yolov13n.pt",
        "pretraining": "COCO detection; LQE and pose branch initialized from scratch",
    },
    "yolo13n-pose-lscd-lqe": {
        "source": CUSTOM_SOURCE,
        "yaml": EXPERIMENT_CONFIGS / "yolo13n-pose-LSCD-LQE.yaml",
        "yaml_source": EXPERIMENT_CONFIGS / "yolo13-pose-LSCD-LQE.yaml",
        "weights": CUSTOM_SOURCE / "yolov13n.pt",
        "pretraining": "COCO detection; LSCD-LQE pose head initialized from scratch",
    },
    "yolo26n-pose": {
        "source": OFFICIAL_SOURCE,
        "yaml": OFFICIAL_SOURCE / "ultralytics/cfg/models/26/yolo26n-pose.yaml",
        "yaml_source": OFFICIAL_SOURCE / "ultralytics/cfg/models/26/yolo26-pose.yaml",
        "weights": ROOT / "yolo26n-pose.pt",
        "pretraining": "COCO pose",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one nano Pose architecture under the LSCD comparison protocol.")
    parser.add_argument("model", choices=MODEL_SPECS)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--pretrained", type=Path, help="Explicit trusted pretrained checkpoint; not a pose-pretrained YOLO13 model")
    parser.add_argument("--scratch", action="store_true", help="No pretrained weights; NOT the paper protocol")
    parser.add_argument("--train-project", type=Path, default=TRAIN_ROOT)
    parser.add_argument("--val-project", type=Path, default=VAL_ROOT)
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def patch_yolov13_checkpoint_namespace() -> None:
    """Map official YOLOv13 checkpoint class names to this local implementation."""
    from ultralytics.nn.extra_modules import yolov13 as y13
    from ultralytics.nn.modules import block, conv

    for name in (
        "AdaHGComputation",
        "AdaHGConv",
        "AdaHyperedgeGen",
        "C3AH",
        "DSBottleneck",
        "DSC3k",
        "DSC3k2",
        "DownsampleConv",
        "FullPAD_Tunnel",
        "FuseModule",
        "HyperACE",
    ):
        setattr(block, name, getattr(y13, name))
    conv.DSConv = y13.DSConv_YOLO13


def best_epoch(results_csv: Path) -> Optional[int]:
    if not results_csv.is_file():
        return None
    with results_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    def fitness(row: dict[str, str]) -> float:
        return 0.1 * (float(row["metrics/mAP50(B)"]) + float(row["metrics/mAP50(P)"])) + 0.9 * (
            float(row["metrics/mAP50-95(B)"]) + float(row["metrics/mAP50-95(P)"])
        )

    return int(max(rows, key=fitness)["epoch"])


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")
    spec = MODEL_SPECS[args.model]
    if args.scratch and args.pretrained:
        raise ValueError("Choose --pretrained or --scratch, not both")
    weights = args.pretrained or spec["weights"]
    require_file(spec["yaml_source"], "model YAML")
    if not args.scratch:
        require_file(weights, "pretrained weights (supply --pretrained)")
    require_file(args.data, "dataset YAML")

    sys.path.insert(0, str(spec["source"]))
    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.model.startswith("yolo13n-pose"):
        patch_yolov13_checkpoint_namespace()

    run_name = args.name or f"{args.model}-seed{args.seed}"
    model = YOLO(str(spec["yaml"]))
    if not args.scratch:
        model.load(str(weights))
    print(f"[source] {ultralytics.__file__}")
    print(f"[model] {args.model}: {type(model.model.model[-1]).__name__}")
    print(f"[pretraining] {'SCRATCH (not paper protocol)' if args.scratch else spec['pretraining']}: {weights}")

    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        cache=False,
        device=args.device,
        workers=args.workers,
        project=str(args.train_project),
        name=run_name,
        exist_ok=args.exist_ok,
        optimizer="auto",
        seed=args.seed,
        deterministic=True,
        close_mosaic=10,
        amp=False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.15,
        degrees=0.0,
        translate=0.05,
        scale=0.1,
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,
        plots=False,
        verbose=True,
    )

    run_dir = Path(model.trainer.save_dir)
    fp32_weights = run_dir / "weights/best_fp32.pt"
    best_weights = fp32_weights if fp32_weights.is_file() else run_dir / "weights/best.pt"
    require_file(best_weights, "best trained weights")

    val_model = YOLO(str(best_weights))
    result = val_model.val(
        data=str(args.data.resolve()),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=0,
        conf=0.001,
        iou=0.7,
        max_det=500,
        device=args.device,
        half=False,
        plots=False,
        project=str(args.val_project),
        name=run_name,
        exist_ok=True,
    )

    summary = {
        "model": args.model,
        "pretraining": "scratch" if args.scratch else spec["pretraining"],
        "pretrained_weights": None if args.scratch else str(weights),
        "best_weights": str(best_weights),
        "best_epoch": best_epoch(run_dir / "results.csv"),
        "parameters": sum(parameter.numel() for parameter in val_model.model.parameters()),
        "ultralytics_version": ultralytics.__version__,
        "ultralytics_file": str(Path(ultralytics.__file__).resolve()),
        "python": sys.executable,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "metrics": {key: float(value) for key, value in result.results_dict.items()},
        "speed_ms_per_image": {key: float(value) for key, value in result.speed.items()},
    }
    output = run_dir / "independent_val_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[summary] {output}")


if __name__ == "__main__":
    main()
