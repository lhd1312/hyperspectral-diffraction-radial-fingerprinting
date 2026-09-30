"""Write a local, single-center Pose dataset configuration without moving data."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            if not (root / kind / split).is_dir():
                raise FileNotFoundError(root / kind / split)
    payload = {"path": root.as_posix(), "train": "images/train", "val": "images/val",
               "names": {0: "particle"}, "nc": 1, "kpt_shape": [1, 3], "flip_idx": [0]}
    if (root / "images/test").is_dir():
        payload["test"] = "images/test"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
