import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from deployment.rk3588.conversion.export_raw_pose_onnx import RawPoseHeadWrapper, use_tanh_gelu
from ultralytics.nn.tasks import PoseModel


def test_raw_stack_onnx_matches_torch(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    root = Path(__file__).resolve().parents[1]
    network = PoseModel(str(root / "experiment_configs/yolo13n-pose-LSCD-LQE.yaml"),
                        nc=1, data_kpt_shape=(1, 3), verbose=False).eval()
    use_tanh_gelu(network)
    wrapper = RawPoseHeadWrapper(network, use_lqe=False, upsample_stack_scales=True).eval()
    data = torch.rand(1, 3, 128, 128, generator=torch.Generator().manual_seed(21))
    path = tmp_path / "raw.onnx"
    kwargs = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    with torch.no_grad():
        expected = wrapper(data)[0]
        torch.onnx.export(wrapper, data, str(path), opset_version=11, input_names=["images"],
                          output_names=["raw"], **kwargs)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {"images": data.numpy()})[0]
    np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-4, atol=1e-4)
