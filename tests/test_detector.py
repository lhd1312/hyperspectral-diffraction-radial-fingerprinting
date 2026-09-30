from pathlib import Path

import pytest
import torch

from ultralytics.nn.tasks import PoseModel

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["vendor/ultralytics/ultralytics/cfg/models/11/yolo11n-pose.yaml",
           "vendor/ultralytics/ultralytics/cfg/models/12/yolo12n-pose.yaml",
           "vendor/ultralytics/ultralytics/cfg/models/13/yolo13n-pose.yaml",
           "experiment_configs/yolo13n-pose-LSCD.yaml",
           "experiment_configs/yolo13n-pose-LQE.yaml",
           "experiment_configs/yolo13n-pose-LSCD-LQE.yaml"]


@pytest.mark.parametrize("config", CONFIGS)
def test_pose_forward(config):
    torch.manual_seed(1)
    model = PoseModel(str(ROOT / config), nc=1, data_kpt_shape=(1, 3), verbose=False).eval()
    with torch.no_grad():
        predictions = model(torch.zeros(1, 3, 128, 128))[0]
    assert predictions.shape == (1, 8, 336)
    assert torch.isfinite(predictions).all()


def test_lscd_lqe_training_loss_backward():
    model = PoseModel(str(ROOT / CONFIGS[-1]), nc=1, data_kpt_shape=(1, 3), verbose=False).train()
    batch = {"img": torch.rand(2, 3, 128, 128), "batch_idx": torch.tensor([0, 1]),
             "cls": torch.zeros(2, 1), "bboxes": torch.tensor([[.5, .5, .2, .2], [.4, .4, .2, .2]]),
             "keypoints": torch.tensor([[[.5, .5, 2.]], [[.4, .4, 2.]]])}
    from ultralytics.cfg import get_cfg
    model.args = get_cfg()
    loss, items = model.loss(batch)
    loss.sum().backward()
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
