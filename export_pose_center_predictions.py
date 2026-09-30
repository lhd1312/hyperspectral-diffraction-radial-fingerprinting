#!/usr/bin/env python3
"""Export deployment-postprocessed pose centers for backend comparison."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Center:
    x: float
    y: float
    box_confidence: float
    keypoint_confidence: float

    @property
    def joint_confidence(self) -> float:
        return min(self.box_confidence, self.keypoint_confidence)


def extract(result, threshold: float, crop_size: int, nms_distance: float):
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    if boxes is None or keypoints is None or len(boxes) == 0:
        return []
    xy = keypoints.xy[:, 0, :].detach().cpu().numpy()
    box_conf = boxes.conf.detach().cpu().numpy()
    keypoint_conf = (
        np.ones_like(box_conf)
        if keypoints.conf is None
        else keypoints.conf[:, 0].detach().cpu().numpy()
    )
    height, width = result.orig_shape
    margin = crop_size / 2.0
    candidates = []
    for point, box_score, keypoint_score in zip(
        xy, box_conf, keypoint_conf
    ):
        x, y = map(float, point)
        values = [x, y, float(box_score), float(keypoint_score)]
        if not np.isfinite(values).all():
            continue
        if box_score < threshold or keypoint_score < threshold:
            continue
        if x < margin or y < margin or x > width - margin or y > height - margin:
            continue
        candidates.append(
            Center(x, y, float(box_score), float(keypoint_score))
        )
    kept = []
    distance_sq = nms_distance**2
    for item in sorted(
        candidates, key=lambda value: value.joint_confidence, reverse=True
    ):
        if all(
            (item.x - other.x) ** 2 + (item.y - other.y) ** 2 >= distance_sq
            for other in kept
        ):
            kept.append(item)
    return sorted(kept, key=lambda value: (value.y, value.x))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ultralytics-source", type=Path)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[768, 1024])
    parser.add_argument("--threshold", type=float, default=0.325)
    parser.add_argument("--pre-nms-conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--crop-size", type=int, default=100)
    parser.add_argument("--center-nms-distance", type=float, default=10.0)
    args = parser.parse_args()

    if args.ultralytics_source:
        package = args.ultralytics_source / "ultralytics" / "__init__.py"
        if package.exists():
            sys.path.insert(0, str(args.ultralytics_source.resolve()))
    from ultralytics import YOLO

    image_paths = sorted(
        path
        for path in args.images.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {args.images}")
    model = YOLO(str(args.model), task="pose")
    rows = []
    for index, path in enumerate(image_paths, start=1):
        result = model.predict(
            source=str(path),
            imgsz=args.imgsz,
            conf=args.pre_nms_conf,
            iou=args.iou,
            max_det=args.max_det,
            rect=False,
            batch=1,
            device=args.device,
            verbose=False,
        )[0]
        centers = extract(
            result,
            args.threshold,
            args.crop_size,
            args.center_nms_distance,
        )
        rows.append(
            {
                "image": path.name,
                "count": len(centers),
                "centers": [
                    {
                        **asdict(center),
                        "joint_confidence": center.joint_confidence,
                    }
                    for center in centers
                ],
            }
        )
        print(f"[{index}/{len(image_paths)}] {path.name}: {len(centers)}")
    payload = {
        "model": str(args.model.resolve()),
        "backend": args.model.suffix.lower().lstrip("."),
        "settings": {
            "imgsz": args.imgsz,
            "threshold": args.threshold,
            "pre_nms_conf": args.pre_nms_conf,
            "iou": args.iou,
            "crop_size": args.crop_size,
            "center_nms_distance": args.center_nms_distance,
        },
        "images": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[PASS] Prediction export: {args.output}")


if __name__ == "__main__":
    main()
