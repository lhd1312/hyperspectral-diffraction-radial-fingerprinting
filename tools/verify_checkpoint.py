"""Verify a trusted study checkpoint against the scoped public YAML implementation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
sys.path.insert(0, str(ROOT / "vendor/ultralytics"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--trusted", action="store_true", help="Acknowledge that loading a .pt can execute Python code")
    args = parser.parse_args()
    if not args.trusted:
        parser.error("Only load trusted artifacts; pass --trusted to acknowledge pickle execution")
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import PoseModel

    torch.set_num_threads(2)
    original = YOLO(str(args.weights)).model.float().eval()
    rebuilt = PoseModel(str(ROOT / "experiment_configs/yolo13n-pose-LSCD-LQE.yaml"),
                        nc=1, data_kpt_shape=(1, 3), verbose=False).eval()
    rebuilt.load_state_dict(original.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(1)
    data = torch.rand((1, 3, 128, 128), generator=generator)
    with torch.no_grad():
        expected, actual = original(data)[0], rebuilt(data)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    print(json.dumps({"strict_state_dict": True, "output_shape": list(actual.shape),
                      "max_absolute_difference": float((actual - expected).abs().max())}))


if __name__ == "__main__":
    main()
