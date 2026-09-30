import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor/ultralytics"))

import torch

torch.set_num_threads(2)
