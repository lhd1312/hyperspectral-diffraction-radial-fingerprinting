# AGPL-3.0. Selected-model parser adapted from the study's Ultralytics fork.
# Channel scaling and graph routing follow that fork, not a new YOLO architecture.
from __future__ import annotations

import ast
from copy import deepcopy

import torch
from torch import nn

from ultralytics.nn.modules import Conv, DWConv, C2f, C3k2, C2PSA, SPPF, Concat, Pose, A2C2f, Detect
from ultralytics.nn.extra_modules.head import Pose_LSCD, Pose_LQE, Pose_LSCD_LQE
from ultralytics.nn.extra_modules.yolov13 import DSC3k2, DSConv_YOLO13, HyperACE, DownsampleConv, FullPAD_Tunnel
from ultralytics.utils import LOGGER
from ultralytics.utils.ops import make_divisible


def parse_model(d, ch, verbose=True, warehouse_manager=None):
    """Build the nano Pose baselines and the three study ablations only."""
    d = deepcopy(d)
    if warehouse_manager is not None or d.get("activation"):
        raise ValueError("Custom warehouse/activation YAMLs are outside the released protocol")
    nc = d["nc"]
    kpt_shape = d.get("kpt_shape", [1, 3])
    scale = d.get("scale") or "n"
    depth, width, max_channels = d["scales"][scale]
    channels, layers, saved = [ch], [], []
    registry = {m.__name__: m for m in (
        Conv, DWConv, C2f, C3k2, C2PSA, SPPF, Concat, Pose, A2C2f, Detect,
        Pose_LSCD, Pose_LQE, Pose_LSCD_LQE, DSC3k2, DSConv_YOLO13,
        HyperACE, DownsampleConv, FullPAD_Tunnel,
    )}
    registry["nn.Upsample"] = nn.Upsample
    for index, (source, repeats, name, args) in enumerate(d["backbone"] + d["head"]):
        if name not in registry:
            raise ValueError(f"Module outside release scope: {name}")
        module = registry[name]
        for j, value in enumerate(args):
            if isinstance(value, str):
                if value == "nc":
                    args[j] = nc
                elif value == "kpt_shape":
                    args[j] = kpt_shape
                else:
                    try:
                        args[j] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        pass
        repeats = max(round(repeats * depth), 1) if repeats > 1 else repeats
        if module in {Conv, DWConv, C2f, C3k2, C2PSA, SPPF, A2C2f, DSC3k2, DSConv_YOLO13}:
            c1, c2 = channels[source], args[0]
            if c2 != nc:
                c2 = make_divisible(min(c2, max_channels) * width, 8)
            args = [c1, c2, *args[1:]]
            if module in {C2f, C3k2, C2PSA, A2C2f, DSC3k2}:
                args.insert(2, repeats)
                repeats = 1
            if module in {C3k2, DSC3k2} and scale in "mlx":
                args[3] = True
            if module is A2C2f and scale in "lx":
                args.extend((True, 1.2))
        elif module is Concat:
            c2 = sum(channels[i] for i in source)
        elif module in {Pose, Pose_LSCD, Pose_LQE, Pose_LSCD_LQE, Detect}:
            args.append([channels[i] for i in source])
            if module in {Pose_LSCD, Pose_LSCD_LQE}:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            c2 = channels[source[0]]
        elif module is HyperACE:
            c1 = channels[source[1]]
            c2 = make_divisible(min(args[0], max_channels) * width, 8)
            edges = int(args[1] * 0.5) if scale == "n" else int(args[1] * 1.5) if scale == "x" else args[1]
            args = [c1, c2, repeats, edges, *args[2:]]
            repeats = 1
            if scale in "lx":
                args.append(False)
        elif module is DownsampleConv:
            c1 = channels[source]
            c2 = c1 * 2
            args = [c1]
            if scale in "lx":
                args.append(False)
                c2 = c1
        elif module is FullPAD_Tunnel:
            c2 = channels[source[0]]
        else:
            c2 = channels[source]
        layer = nn.Sequential(*(module(*args) for _ in range(repeats))) if repeats > 1 else module(*args)
        layer.np = sum(p.numel() for p in layer.parameters())
        layer.i, layer.f, layer.type = index, source, str(module)[8:-2]
        if verbose:
            LOGGER.info(f"{index:>3} {str(source):>16} {layer.np:>10} {name:<24} {args}")
        saved.extend(i % index for i in ([source] if isinstance(source, int) else source) if i != -1)
        layers.append(layer)
        if index == 0:
            channels = []
        channels.append(c2)
    return nn.Sequential(*layers), sorted(saved)
