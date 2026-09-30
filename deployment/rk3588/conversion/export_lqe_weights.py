#!/usr/bin/env python3
"""Export the tiny LQE branches from a trained pose model to NumPy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ultralytics-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.weights = args.weights.resolve()
    args.ultralytics_source = args.ultralytics_source.resolve()
    args.output = args.output.resolve()
    sys.path.insert(0, str(args.ultralytics_source))

    from ultralytics import YOLO

    model = YOLO(str(args.weights)).model.float().eval()
    head = model.model[-1]
    if not hasattr(head, "lqe") or len(head.lqe) != 3:
        raise TypeError(f"Expected a three-scale LQE pose head, received {type(head).__name__}")

    arrays: dict[str, np.ndarray] = {}
    for index, branch in enumerate(head.lqe):
        layers = branch.reg_conf.layers
        if len(layers) != 2:
            raise TypeError(f"Expected two LQE layers at scale {index}, received {len(layers)}")
        arrays[f"scale_{index}_weight_0"] = layers[0].weight.detach().cpu().numpy()[:, :, 0, 0]
        arrays[f"scale_{index}_bias_0"] = layers[0].bias.detach().cpu().numpy()
        arrays[f"scale_{index}_weight_1"] = layers[1].weight.detach().cpu().numpy()[:, :, 0, 0]
        arrays[f"scale_{index}_bias_1"] = layers[1].bias.detach().cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **arrays)
    metadata = {
        "source_weights": str(args.weights),
        "pose_head": type(head).__name__,
        "strides": [float(value) for value in head.stride.detach().cpu().tolist()],
        "reg_max": int(head.reg_max),
        "top_k": int(head.lqe[0].k),
        "hidden_dim": int(arrays["scale_0_weight_0"].shape[0]),
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[PASS] LQE weights: {args.output}")


if __name__ == "__main__":
    main()
