import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.optimize import linear_sum_assignment as scipy_assignment

import pinn_diffusion_hsi_augmentation as aug
from analyze_measured_calibration import leave_one_out_predictions, ols_fit
from convert_fixed_boxes_to_pose import convert_label
from deployment.rk3588.tools.evaluate_rknn_pose_batch import linear_sum_assignment, match_with_radius
from evaluate_augmentation_classification import exact_mcnemar_pvalue, holm_adjust


def test_envi_roundtrip(tmp_path):
    cube = np.random.default_rng(1).random((8, 24, 32)).astype(np.float32)
    meta = aug.EnviMeta(32, 24, 8, 4, "bsq", 0, np.linspace(411, 779, 8))
    path = tmp_path / "test.hdr"
    aug.write_envi_cube(path, cube, meta, "ARTIFICIAL TEST ONLY")
    actual_meta, actual_cube = aug.read_envi_cube(path)
    np.testing.assert_array_equal(actual_cube, cube)
    np.testing.assert_allclose(actual_meta.wavelengths, meta.wavelengths, atol=0.0001)


def test_discovery_excludes_test_directories(tmp_path):
    for _, folder, _ in aug.CATEGORY_SPECS:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "sample_1.hdr").touch()
        (tmp_path / (folder + "_test")).mkdir()
        (tmp_path / (folder + "_test") / "sample_2.hdr").touch()
    records = aug.discover_records(tmp_path)
    assert len(records) == 5
    assert all(r.hdr_path.stem == "sample_1" for r in records)


@pytest.mark.parametrize("indices,error", [([3], IndexError), ([-1], IndexError), ([0, 0], ValueError),
                                         ([], ValueError), ([0.2], ValueError), ([True], ValueError)])
def test_invalid_split_rejected(tmp_path, indices, error):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": indices}), encoding="utf-8")
    with pytest.raises(error):
        aug.select_fit_records([object(), object()], path, "train")


def test_missing_split_is_not_all_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        aug.select_fit_records([object()], tmp_path / "missing.json", "train")


def test_training_folder_policy():
    records = [object(), object()]
    assert aug.select_fit_records(records, None, "train") is records
    with pytest.raises(ValueError):
        aug.select_fit_records(records, None, "test")


def test_autoencoder_and_denoiser_backward():
    torch.manual_seed(1)
    ae = aug.CubeAutoencoder(8, 32, 32, 5, latent_dim=8, base_channels=8)
    data = torch.rand(2, 8, 32, 32)
    labels = torch.tensor([0, 4])
    reconstruction, latent = ae(data, labels)
    assert reconstruction.shape == data.shape
    noise = aug.LatentDenoiser(8, 5, hidden_dim=32)(latent, torch.tensor([0, 3]), labels)
    loss = (reconstruction - data).square().mean() + noise.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in ae.parameters() if p.grad is not None)


def test_exact_linear_calibration_and_lopo():
    x = np.arange(1, 7, dtype=float)
    y = 100 * x + 7
    fit = ols_fit(x, y)
    assert fit["slope"] == pytest.approx(100)
    assert fit["intercept"] == pytest.approx(7)
    np.testing.assert_allclose(leave_one_out_predictions(x, y), y)
    with pytest.raises(ValueError):
        ols_fit(np.ones(6), y)


def test_paired_statistics():
    assert exact_mcnemar_pvalue(0, 0) == 1
    assert exact_mcnemar_pvalue(10, 0) == pytest.approx(2 / 2**10)
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


@pytest.mark.parametrize("shape", [(3, 4), (4, 3), (5, 5), (0, 3)])
def test_assignment_matches_scipy(shape):
    costs = np.random.default_rng(21).random(shape)
    i, j = linear_sum_assignment(costs)
    si, sj = scipy_assignment(costs)
    assert costs[i, j].sum() == pytest.approx(costs[si, sj].sum())


def test_radius_matching_ignores_invalid_pairs():
    assert match_with_radius(np.empty((0, 2)), 10) == []
    assert match_with_radius(np.array([[2., 20.], [20., 4.]]), 10) == [(0, 0, 2.), (1, 1, 4.)]


def test_fixed_center_conversion_rejects_boundary(tmp_path):
    source, output = tmp_path / "input.txt", tmp_path / "output.txt"
    source.write_text("0 0.5 0.5 0.1 0.1\n0 0.01 0.5 0.1 0.1\n", encoding="utf-8")
    assert convert_label(source, output) == (1, 1)
    columns = output.read_text().split()
    assert len(columns) == 8
    assert columns[1:3] == columns[5:7]
