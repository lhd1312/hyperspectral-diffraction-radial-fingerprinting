#!/usr/bin/env python3
"""Export the frozen custom pose detector to a fixed-shape ONNX graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import onnx


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = (
    ROOT
    / "jetson_predeployment_package"
    / "models"
    / "yolo13n_pose_lscd_lqe_best.pt"
)
DEFAULT_SOURCE = ROOT / "vendor" / "ultralytics"
DEFAULT_OUTPUT = ROOT / "jetson_deployment" / "artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_shape(value_info) -> list[object]:
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        shape.append(dim.dim_value if dim.dim_value else dim.dim_param)
    return shape


def validate_onnx(path: Path, height: int, width: int) -> dict[str, object]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    metadata = {item.key: item.value for item in model.metadata_props}
    inputs = {item.name: tensor_shape(item) for item in model.graph.input}
    outputs = {item.name: tensor_shape(item) for item in model.graph.output}
    if inputs.get("images") != [1, 3, height, width]:
        raise RuntimeError(f"Unexpected ONNX input shape: {inputs}")
    if metadata.get("task") != "pose" or metadata.get("kpt_shape") != "[1, 3]":
        raise RuntimeError(f"Pose metadata is incomplete: {metadata}")
    return {
        "ir_version": int(model.ir_version),
        "opsets": [
            {"domain": item.domain, "version": int(item.version)}
            for item in model.opset_import
        ],
        "inputs": inputs,
        "outputs": outputs,
        "node_count": len(model.graph.node),
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--ultralytics-source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="0")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--reuse", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if not (args.ultralytics_source / "ultralytics" / "__init__.py").exists():
        raise FileNotFoundError(args.ultralytics_source)
    if args.height % 32 or args.width % 32:
        raise ValueError("ONNX height and width must be multiples of stride 32.")

    staged_pt = (
        args.output_dir
        / f"yolo13n_pose_lscd_lqe_{args.height}x{args.width}_fp32.pt"
    )
    onnx_path = staged_pt.with_suffix(".onnx")
    if not args.reuse or not onnx_path.exists():
        shutil.copy2(args.weights, staged_pt)
        source_text = str(args.ultralytics_source.resolve())
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        from ultralytics import YOLO

        model = YOLO(str(staged_pt))
        exported = Path(
            model.export(
                format="onnx",
                imgsz=(args.height, args.width),
                batch=1,
                dynamic=False,
                simplify=args.simplify,
                opset=args.opset,
                half=False,
                device=args.device,
            )
        )
        if exported.resolve() != onnx_path.resolve():
            shutil.copy2(exported, onnx_path)

    graph = validate_onnx(onnx_path, args.height, args.width)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_weights": str(args.weights.resolve()),
        "source_sha256": sha256(args.weights),
        "onnx_path": str(onnx_path.resolve()),
        "onnx_sha256": sha256(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "export": {
            "imgsz": [args.height, args.width],
            "batch": 1,
            "dynamic": False,
            "fp32": True,
            "opset": args.opset,
            "simplify": args.simplify,
        },
        "graph": graph,
    }
    summary_path = args.output_dir / "onnx_export_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[PASS] ONNX export: {onnx_path}")
    print(f"  input: {graph['inputs']}")
    print(f"  output: {graph['outputs']}")
    print(f"  sha256: {summary['onnx_sha256']}")


if __name__ == "__main__":
    main()
