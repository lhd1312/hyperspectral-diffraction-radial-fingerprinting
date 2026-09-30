# Third-Party Sources and License Review

This release includes selected local source snapshots, not a claim of authorship over upstream networks.

| Location | Provenance | Notice |
| --- | --- | --- |
| `vendor/ultralytics` | Study's local modified Ultralytics 8.3.9 tree | Upstream AGPL-3.0 LICENSE and file notices retained |
| `vendor/ultralytics-official` | Study's local official Ultralytics tree used for v8/26 baselines | Its own upstream LICENSE/README/metadata retained |
| `vendor/ultralytics/ultralytics/nn/extra_modules/yolov13.py` | YOLO13 module from the local research fork | Original source retained; no new authorship asserted |
| `vendor/ultralytics/ultralytics/nn/extra_modules/head.py` | Selected LSCD/LQE classes from the research fork | Original selected class bodies retained; unrelated experimental extensions removed |

The scoped parser/registry and dependency cleanup are described in `docs/CHANGELOG.md`; unrelated upstream model families are intentionally unsupported. The exact upstream Git commits were not recorded in the local source snapshots.

Upstream license texts and existing file-level notices are retained. The selected LSCD/LQE and YOLO13 implementations are attributed to the local research fork rather than claimed as newly authored modules.

External runtimes (PyTorch, scikit-learn, ONNX, TensorRT, RKNN Toolkit2/Lite2, OpenCV, etc.) keep their own licenses and are not bundled as wheels. Download them through their respective distribution channels. Model checkpoints and experimental data require separate provenance and sharing decisions.
