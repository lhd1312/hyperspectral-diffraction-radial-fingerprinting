#!/usr/bin/env python3
"""Export YOLO13 pose features without Ultralytics decode/NMS operators.

Originally developed for the study's earlier RKNN export, this raw-head layout
is also used by the RK3588 Toolkit2 conversion. It keeps the trained backbone,
neck, LSCD head and optionally LQE in ONNX. The deployed stacked variant moves
LQE, DFL/anchor/keypoint decode and NMS to the host CPU.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
from torch import nn


class RawPoseHeadWrapper(nn.Module):
    """Run the model through its raw three-scale pose prediction heads."""

    def __init__(
        self,
        network: nn.Module,
        use_lqe: bool = True,
        combine_scale_outputs: bool = False,
        flatten_concat_scales: bool = False,
        mosaic_scales: bool = False,
        upsample_stack_scales: bool = False,
        scale_indices: tuple[int, ...] = (0, 1, 2),
    ) -> None:
        super().__init__()
        self.network = network
        self.use_lqe = use_lqe
        self.combine_scale_outputs = combine_scale_outputs
        self.flatten_concat_scales = flatten_concat_scales
        self.mosaic_scales = mosaic_scales
        self.upsample_stack_scales = upsample_stack_scales
        self.scale_indices = scale_indices

    def forward(self, images: torch.Tensor):
        saved = []
        x = images
        layers = self.network.model

        for module in layers[:-1]:
            if module.f != -1:
                x = (
                    saved[module.f]
                    if isinstance(module.f, int)
                    else [x if index == -1 else saved[index] for index in module.f]
                )
            x = module(x)
            saved.append(x if module.i in self.network.save else None)

        head = layers[-1]
        features = (
            saved[head.f]
            if isinstance(head.f, int)
            else [x if index == -1 else saved[index] for index in head.f]
        )

        outputs = []
        for index in self.scale_indices:
            keypoints = head.cv4[index](features[index])
            shared = head.share_conv(head.conv[index](features[index]))
            corners = head.scale[index](head.cv2(shared))
            scores = head.cv3(shared)
            if self.use_lqe:
                scores = head.lqe[index](scores, corners)
            if (
                self.combine_scale_outputs
                or self.flatten_concat_scales
                or self.mosaic_scales
                or self.upsample_stack_scales
            ):
                outputs.append(torch.cat((corners, scores, keypoints), dim=1))
            else:
                outputs.extend((corners, scores, keypoints))
        if self.flatten_concat_scales:
            return (torch.cat([tensor.flatten(2) for tensor in outputs], dim=2),)
        if self.mosaic_scales:
            target_width = outputs[0].shape[3]
            padded = [
                torch.nn.functional.pad(tensor, (0, target_width - tensor.shape[3], 0, 0))
                for tensor in outputs
            ]
            return (torch.cat(padded, dim=2),)
        if self.upsample_stack_scales:
            target_size = outputs[0].shape[2:]
            expanded = [
                tensor
                if tensor.shape[2:] == target_size
                else torch.nn.functional.interpolate(tensor, size=target_size, mode="nearest")
                for tensor in outputs
            ]
            return (torch.cat(expanded, dim=1),)
        return tuple(outputs)


def use_tanh_gelu(module: nn.Module) -> int:
    """Replace weightless exact GELU modules with the standard tanh approximation."""

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GELU) and child.approximate == "none":
            setattr(module, name, nn.GELU(approximate="tanh"))
            replaced += 1
        else:
            replaced += use_tanh_gelu(child)
    return replaced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--ultralytics-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument(
        "--without-lqe",
        action="store_true",
        help="Diagnostic export that omits LQE but retains the LSCD head.",
    )
    parser.add_argument(
        "--approximate-gelu",
        action="store_true",
        help="Replace exact GELU/Erf with the standard tanh approximation.",
    )
    parser.add_argument(
        "--combine-scale-outputs",
        action="store_true",
        help="Expose one 68-channel tensor per scale instead of nine outputs.",
    )
    parser.add_argument(
        "--strides",
        type=int,
        nargs="+",
        choices=(8, 16, 32),
        default=(8, 16, 32),
        help="Prediction scales to expose. Useful for RKNN compatibility tests.",
    )
    parser.add_argument(
        "--flatten-concat-scales",
        action="store_true",
        help="Expose every scale as one fixed [1, 68, anchors] output tensor.",
    )
    parser.add_argument(
        "--mosaic-scales",
        action="store_true",
        help="Pack scales into one padded 4-D tensor without reshape operators.",
    )
    parser.add_argument(
        "--upsample-stack-scales",
        action="store_true",
        help="Nearest-upsample every scale and concatenate into one 4-D tensor.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.weights = args.weights.resolve()
    args.ultralytics_source = args.ultralytics_source.resolve()
    args.output = args.output.resolve()
    if not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if not (args.ultralytics_source / "ultralytics" / "__init__.py").exists():
        raise FileNotFoundError(args.ultralytics_source)
    if args.height % 32 or args.width % 32:
        raise ValueError("Height and width must be multiples of stride 32.")

    sys.path.insert(0, str(args.ultralytics_source))
    from ultralytics import YOLO

    network = YOLO(str(args.weights)).model.float().eval()
    gelu_replacements = use_tanh_gelu(network) if args.approximate_gelu else 0
    head = network.model[-1]
    if not all(hasattr(head, name) for name in ("conv", "share_conv", "cv2", "cv3", "cv4", "lqe")):
        raise TypeError(f"Unsupported pose head: {type(head).__name__}")

    stride_to_index = {8: 0, 16: 1, 32: 2}
    selected_strides = tuple(args.strides)
    scale_indices = tuple(stride_to_index[stride] for stride in selected_strides)
    wrapper = RawPoseHeadWrapper(
        network,
        use_lqe=not args.without_lqe,
        combine_scale_outputs=args.combine_scale_outputs,
        flatten_concat_scales=args.flatten_concat_scales,
        mosaic_scales=args.mosaic_scales,
        upsample_stack_scales=args.upsample_stack_scales,
        scale_indices=scale_indices,
    ).eval()
    example = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
    if args.upsample_stack_scales:
        output_names = ["pose_raw_stack"]
    elif args.mosaic_scales:
        output_names = ["pose_raw_mosaic"]
    elif args.flatten_concat_scales:
        output_names = ["pose_raw_all"]
    elif args.combine_scale_outputs:
        output_names = [f"pose_raw_s{stride}" for stride in selected_strides]
    else:
        output_names = [
            f"{kind}_s{stride}"
            for stride in selected_strides
            for kind in ("dfl", "score", "keypoint")
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        raw = wrapper(example)
        torch.onnx.export(
            wrapper,
            example,
            str(args.output),
            input_names=["images"],
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=None,
            **({"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}),
        )

    summary = {
        "weights": str(args.weights),
        "onnx": str(args.output),
        "input_shape": [1, 3, args.height, args.width],
        "opset": args.opset,
        "lqe_in_graph": not args.without_lqe,
        "gelu_approximation": "tanh" if args.approximate_gelu else "none",
        "gelu_replacements": gelu_replacements,
        "combine_scale_outputs": args.combine_scale_outputs,
        "flatten_concat_scales": args.flatten_concat_scales,
        "mosaic_scales": args.mosaic_scales,
        "upsample_stack_scales": args.upsample_stack_scales,
        "strides": list(selected_strides),
        "reg_max": int(head.reg_max),
        "classes": int(head.nc),
        "keypoint_shape": list(head.kpt_shape),
        "outputs": {
            name: list(tensor.shape) for name, tensor in zip(output_names, raw)
        },
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[PASS] Raw pose ONNX: {args.output}")
    for name, tensor in zip(output_names, raw):
        print(f"  {name}: {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
