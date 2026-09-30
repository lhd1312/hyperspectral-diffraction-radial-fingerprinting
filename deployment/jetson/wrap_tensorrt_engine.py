#!/usr/bin/env python3
"""Prepend Ultralytics metadata to a raw trtexec TensorRT engine."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def typed_metadata(
    onnx_path: Path, metadata_json: Path | None
) -> dict[str, object]:
    if metadata_json is not None:
        payload = json.loads(metadata_json.read_text(encoding="utf-8"))
        metadata = payload.get("graph", {}).get("metadata", payload)
    else:
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError(
                "Install onnx or pass --metadata-json onnx_export_summary.json"
            ) from exc
        model = onnx.load(str(onnx_path), load_external_data=False)
        metadata = {item.key: item.value for item in model.metadata_props}
    for key in ("stride", "batch"):
        if key in metadata:
            metadata[key] = int(metadata[key])
    for key in ("imgsz", "names", "kpt_shape"):
        if key in metadata:
            metadata[key] = ast.literal_eval(metadata[key])
    required = {"stride", "task", "batch", "imgsz", "names", "kpt_shape"}
    missing = sorted(required - metadata.keys())
    if missing:
        raise RuntimeError(f"Missing ONNX metadata: {missing}")
    if metadata["task"] != "pose" or metadata["kpt_shape"] != [1, 3]:
        raise RuntimeError(f"Unexpected pose metadata: {metadata}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--raw-engine", type=Path, required=True)
    parser.add_argument("--output-engine", type=Path, required=True)
    args = parser.parse_args()
    if args.raw_engine.resolve() == args.output_engine.resolve():
        raise ValueError("Raw and wrapped engine paths must be different.")
    metadata = typed_metadata(args.onnx, args.metadata_json)
    encoded = json.dumps(metadata, ensure_ascii=True).encode("utf-8")
    args.output_engine.parent.mkdir(parents=True, exist_ok=True)
    with args.output_engine.open("wb") as output:
        output.write(len(encoded).to_bytes(4, byteorder="little", signed=True))
        output.write(encoded)
        with args.raw_engine.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                output.write(block)
    summary = {
        "onnx": str(args.onnx.resolve()),
        "onnx_sha256": sha256(args.onnx),
        "raw_engine": str(args.raw_engine.resolve()),
        "raw_engine_sha256": sha256(args.raw_engine),
        "wrapped_engine": str(args.output_engine.resolve()),
        "wrapped_engine_sha256": sha256(args.output_engine),
        "metadata_bytes": len(encoded),
        "metadata": metadata,
    }
    summary_path = args.output_engine.with_suffix(".metadata.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[PASS] Wrapped TensorRT engine: {args.output_engine}")
    print(f"  metadata bytes: {len(encoded)}")
    print(f"  sha256: {summary['wrapped_engine_sha256']}")


if __name__ == "__main__":
    main()
