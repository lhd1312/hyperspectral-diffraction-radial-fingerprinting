#!/usr/bin/env python3
"""Convert the raw stacked YOLO13 pose ONNX graph for RK3588."""

from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def check(result: int | None, stage: str) -> None:
    if result not in (0, None):
        raise RuntimeError(f"RKNN {stage} failed with code {result}")


def main() -> None:
    args = parse_args()
    from rknn.api import RKNN
    if args.int8 and args.dataset is None:
        raise ValueError("--dataset is required with --int8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runtime = RKNN(verbose=args.verbose)
    try:
        check(
            runtime.config(
                mean_values=[[0.0, 0.0, 0.0]],
                std_values=[[255.0, 255.0, 255.0]],
                target_platform="rk3588",
            ),
            "config",
        )
        check(runtime.load_onnx(model=str(args.onnx)), "load_onnx")
        build_arguments = {"do_quantization": bool(args.int8)}
        if args.int8:
            build_arguments["dataset"] = str(args.dataset)
        check(runtime.build(**build_arguments), "build")
        check(runtime.export_rknn(str(args.output)), "export_rknn")
    finally:
        runtime.release()

    try:
        toolkit_version = metadata.version("rknn-toolkit2")
    except metadata.PackageNotFoundError:
        toolkit_version = "unknown"
    conversion_metadata = {
        "source_onnx": str(args.onnx.resolve()),
        "target_platform": "rk3588",
        "compiler_toolkit_version": toolkit_version,
        "quantization": "INT8" if args.int8 else "FP16",
        "output_rknn": str(args.output.resolve()),
        "input_preprocessing": {"color": "RGB", "mean": [0, 0, 0], "std": [255, 255, 255]},
        "expected_output": [1, 204, 96, 128],
        "runtime_family": "RKNN Toolkit2 / Lite2 for RK3588",
    }
    args.output.with_suffix(".conversion.json").write_text(
        json.dumps(conversion_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(conversion_metadata, indent=2))


if __name__ == "__main__":
    main()
