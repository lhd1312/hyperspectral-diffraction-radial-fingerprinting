from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEW_ROOT = ROOT / "yolo_hsi_detection_review_yolo26"
DEFAULT_OUTPUT_ROOT = ROOT / "yolo_hsi_pose_center_dataset"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert trusted fixed YOLO boxes to one-center-keypoint pose labels.")
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--source-label-dir", type=str, default="labels_original")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def materialize_image(source: Path, target: Path, copy_images: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if not copy_images:
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def convert_label(source: Path, target: Path) -> tuple[int, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        target.write_text("", encoding="utf-8")
        return 0, 0
    output_lines = []
    rejected = 0
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Expected 5 columns at {source}:{line_number}, got {len(parts)}")
        _, x_text, y_text, w_text, h_text = parts
        x, y, width, height = map(float, (x_text, y_text, w_text, h_text))
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            rejected += 1
            continue
        if x - width / 2.0 < -1e-6 or x + width / 2.0 > 1.0 + 1e-6:
            rejected += 1
            continue
        if y - height / 2.0 < -1e-6 or y + height / 2.0 > 1.0 + 1e-6:
            rejected += 1
            continue
        output_lines.append(
            f"0 {x:.8f} {y:.8f} {width:.8f} {height:.8f} {x:.8f} {y:.8f} 2"
        )
    target.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    return len(output_lines), rejected


def main() -> None:
    args = build_arg_parser().parse_args()
    review_root = args.review_root.resolve()
    output_root = args.output_root.resolve()
    source_labels_root = review_root / args.source_label_dir
    if not source_labels_root.exists():
        raise SystemExit(f"Missing source labels: {source_labels_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    split_stats = {}
    for split in ("train", "val", "test"):
        source_image_dir = review_root / "images" / split
        if not source_image_dir.exists():
            continue
        images = sorted(path for path in source_image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = 0
        rejected = 0
        nonempty = 0
        for image_path in images:
            target_image = output_root / "images" / split / image_path.name
            target_label = output_root / "labels" / split / f"{image_path.stem}.txt"
            if target_label.exists() and not args.overwrite:
                raise SystemExit(f"Output label already exists; use --overwrite to regenerate: {target_label}")
            materialize_image(image_path, target_image, args.copy_images)
            converted, rejected_here = convert_label(
                source_labels_root / split / f"{image_path.stem}.txt",
                target_label,
            )
            labels += converted
            rejected += rejected_here
            nonempty += int(converted > 0)
        split_stats[split] = {
            "images": len(images),
            "nonempty_labels": nonempty,
            "pose_instances": labels,
            "rejected_out_of_bounds": rejected,
        }

    data_yaml = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: ['particle']\n"
        "kpt_shape: [1, 3]\n"
        "flip_idx: [0]\n"
        "kpt_names:\n"
        "  0: ['diffraction_center']\n"
    )
    (output_root / "data.yaml").write_text(data_yaml, encoding="utf-8")
    (output_root / "classes.txt").write_text("particle\n", encoding="utf-8")
    metadata = {
        "review_root": str(review_root),
        "source_labels": str(source_labels_root),
        "source_policy": (
            "Reviewed labels used as final ground truth."
            if args.source_label_dir == "labels"
            else f"Pose labels converted from {args.source_label_dir}."
        ),
        "pose_label_format": "class box_x box_y box_w box_h keypoint_x keypoint_y visibility",
        "keypoint_definition": "Fixed-box center, visibility=2",
        "splits": split_stats,
    }
    (output_root / "conversion_info.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[output] {output_root}")


if __name__ == "__main__":
    main()
