from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "yolo_hsi_detection_review_yolo26"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ReviewBox:
    x: int
    y: int
    source: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review fixed-box YOLO26 prelabels with manual/candidate colors.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="all")
    parser.add_argument("--box-size", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-display-width", type=int, default=1500)
    parser.add_argument("--max-display-height", type=int, default=950)
    parser.add_argument("--include-reviewed", action="store_true")
    parser.add_argument("--only-candidates", action="store_true", help="Skip images that have no class-1 candidates.")
    parser.add_argument(
        "--order",
        choices=["candidates-desc", "name"],
        default="candidates-desc",
        help="Review the images with the most new model candidates first by default.",
    )
    return parser


def image_label_pairs(dataset_root: Path, split_choice: str) -> list[tuple[str, Path, Path]]:
    splits = [split_choice] if split_choice != "all" else ["train", "val", "test"]
    pairs = []
    for split in splits:
        image_dir = dataset_root / "images" / split
        if not image_dir.exists():
            continue
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
            pairs.append((split, image_path, dataset_root / "labels" / split / f"{image_path.stem}.txt"))
    return pairs


def load_boxes(label_path: Path, width: int, height: int) -> list[ReviewBox]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        source = 1 if int(parts[0]) == 1 else 0
        boxes.append(ReviewBox(int(round(float(parts[1]) * width)), int(round(float(parts[2]) * height)), source))
    return boxes


def write_boxes(label_path: Path, boxes: list[ReviewBox], width: int, height: int, box_size: int, approve: bool) -> None:
    half = box_size // 2
    lines = []
    for box in boxes:
        box.x = int(np.clip(box.x, half, width - half))
        box.y = int(np.clip(box.y, half, height - half))
        class_id = 0 if approve else box.source
        lines.append(
            f"{class_id} {box.x / width:.8f} {box.y / height:.8f} "
            f"{box_size / width:.8f} {box_size / height:.8f}"
        )
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def nearest_box(boxes: list[ReviewBox], x: int, y: int, max_distance: float) -> int:
    if not boxes:
        return -1
    distances = [((box.x - x) ** 2 + (box.y - y) ** 2) ** 0.5 for box in boxes]
    index = int(np.argmin(distances))
    return index if distances[index] <= max_distance else -1


def draw_view(image: np.ndarray, boxes: list[ReviewBox], selected: int, box_size: int, scale: float, title: str) -> np.ndarray:
    view = cv2.resize(
        image,
        (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    half = box_size / 2.0
    for index, box in enumerate(boxes):
        color = (0, 180, 0) if box.source == 0 else (0, 210, 255)
        if index == selected:
            color = (0, 60, 255)
        x0 = int(round((box.x - half) * scale))
        y0 = int(round((box.y - half) * scale))
        x1 = int(round((box.x + half) * scale))
        y1 = int(round((box.y + half) * scale))
        cv2.rectangle(view, (x0, y0), (x1, y1), color, 2)
        cv2.circle(view, (int(round(box.x * scale)), int(round(box.y * scale))), 3, color, -1)
    manual = sum(box.source == 0 for box in boxes)
    candidates = sum(box.source == 1 for box in boxes)
    help_lines = [
        title,
        f"green/manual={manual}  yellow/candidate={candidates}  total={len(boxes)}",
        "left add/move | right delete | a approve candidates | r remove candidates",
        "s approve+save next | b previous | u undo | c clear | q save+quit",
    ]
    y = 24
    for line in help_lines:
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return view


def load_reviewed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def save_reviewed(path: Path, reviewed: set[str]) -> None:
    path.write_text("\n".join(sorted(reviewed)) + ("\n" if reviewed else ""), encoding="utf-8")


def load_candidate_counts(dataset_root: Path) -> dict[str, int]:
    summary_path = dataset_root / "candidate_summary.csv"
    if not summary_path.exists():
        return {}
    counts = {}
    with summary_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            counts[f"{row['split']}/{row['image']}"] = int(row.get("new_candidates", 0))
    return counts


def review_one(
    image_path: Path,
    label_path: Path,
    title: str,
    box_size: int,
    max_display_width: int,
    max_display_height: int,
) -> str:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"[warning] Cannot read {image_path}")
        return "next"
    height, width = image.shape[:2]
    scale = min(max_display_width / width, max_display_height / height, 1.0)
    boxes = load_boxes(label_path, width, height)
    selected = -1
    history: list[list[ReviewBox]] = []
    window_name = "YOLO26 label review"

    def snapshot() -> None:
        history.append([ReviewBox(box.x, box.y, box.source) for box in boxes])
        if len(history) > 60:
            history.pop(0)

    def mouse_callback(event: int, x_view: int, y_view: int, _flags: int, _param: object) -> None:
        nonlocal selected
        x = int(round(x_view / scale))
        y = int(round(y_view / scale))
        half = box_size // 2
        x = int(np.clip(x, half, width - half))
        y = int(np.clip(y, half, height - half))
        if event == cv2.EVENT_LBUTTONDOWN:
            index = nearest_box(boxes, x, y, box_size * 0.42)
            snapshot()
            if index >= 0:
                boxes[index] = ReviewBox(x, y, 0)
                selected = index
            else:
                boxes.append(ReviewBox(x, y, 0))
                selected = len(boxes) - 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            index = nearest_box(boxes, x, y, box_size)
            if index >= 0:
                snapshot()
                boxes.pop(index)
                selected = -1

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    while True:
        cv2.imshow(window_name, draw_view(image, boxes, selected, box_size, scale, title))
        key = cv2.waitKey(30) & 0xFF
        if key in (255, 0xFF):
            continue
        if key == ord("s"):
            write_boxes(label_path, boxes, width, height, box_size, approve=True)
            return "approved"
        if key == ord("b"):
            write_boxes(label_path, boxes, width, height, box_size, approve=False)
            return "previous"
        if key == ord("u") and history:
            boxes[:] = history.pop()
            selected = -1
        elif key == ord("a"):
            snapshot()
            for box in boxes:
                box.source = 0
        elif key == ord("r"):
            snapshot()
            boxes[:] = [box for box in boxes if box.source == 0]
            selected = -1
        elif key == ord("c"):
            snapshot()
            boxes.clear()
            selected = -1
        elif key in (ord("q"), 27):
            write_boxes(label_path, boxes, width, height, box_size, approve=False)
            return "quit"


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    pairs = image_label_pairs(dataset_root, args.split)
    if not pairs:
        raise SystemExit(f"No images found under {dataset_root}")
    progress_path = dataset_root / "reviewed.txt"
    reviewed = load_reviewed(progress_path)
    candidate_counts = load_candidate_counts(dataset_root)
    if not args.include_reviewed:
        pairs = [pair for pair in pairs if f"{pair[0]}/{pair[1].name}" not in reviewed]
    if args.only_candidates:
        pairs = [pair for pair in pairs if candidate_counts.get(f"{pair[0]}/{pair[1].name}", 0) > 0]
    if args.order == "candidates-desc":
        pairs.sort(key=lambda pair: (-candidate_counts.get(f"{pair[0]}/{pair[1].name}", 0), pair[0], pair[1].name))
    if not pairs:
        print("All selected images have already been reviewed.")
        return

    index = min(max(args.start_index, 0), len(pairs) - 1)
    while 0 <= index < len(pairs):
        split, image_path, label_path = pairs[index]
        key = f"{split}/{image_path.name}"
        action = review_one(
            image_path,
            label_path,
            f"{index + 1}/{len(pairs)}  {key}",
            args.box_size,
            args.max_display_width,
            args.max_display_height,
        )
        if action == "approved":
            reviewed.add(key)
            save_reviewed(progress_path, reviewed)
            index += 1
        elif action == "previous":
            index = max(0, index - 1)
        elif action == "quit":
            break
        else:
            index += 1
    cv2.destroyAllWindows()
    print(f"[progress] reviewed={len(reviewed)} file={progress_path}")


if __name__ == "__main__":
    main()
