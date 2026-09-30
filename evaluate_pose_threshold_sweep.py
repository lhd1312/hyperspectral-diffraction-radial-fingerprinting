from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "yolo_hsi_pose_runs" / "hsi_particle_yolo26n_pose_center_green_20260719"


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep box/keypoint confidence for center matching and counting.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_RUN / "weights" / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "yolo_hsi_pose_center_dataset" / "data.yaml")
    parser.add_argument("--ultralytics-source", type=Path, default=ROOT / "vendor" / "ultralytics")
    parser.add_argument("--output", type=Path, default=DEFAULT_RUN / "threshold_sweep.csv")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
    args = parser.parse_args()

    source_root = args.ultralytics_source.resolve()
    sys.path.insert(0, str(source_root))
    from ultralytics import YOLO
    from train_yolo26_pose_center import center_metrics

    model = YOLO(str(args.weights))
    rows = []
    for threshold in args.thresholds:
        summary, _ = center_metrics(
            model,
            args.data,
            args.imgsz,
            args.device,
            threshold,
            threshold,
            20.0,
        )
        row = {
            "threshold": threshold,
            "predicted_centers": summary["predicted_centers"],
            "precision_5px": summary["precision_at_5px"],
            "recall_5px": summary["recall_at_5px"],
            "f1_5px": summary["f1_at_5px"],
            "precision_10px": summary["precision_at_10px"],
            "recall_10px": summary["recall_at_10px"],
            "f1_10px": summary["f1_at_10px"],
            "precision_20px": summary["precision_at_20px"],
            "recall_20px": summary["recall_at_20px"],
            "f1_20px": summary["f1_at_20px"],
            "center_mae_px": summary["center_mae_px_matched"],
            "count_mae_per_image": summary["count_mae_per_image"],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[output] {args.output}")


if __name__ == "__main__":
    main()
