# Source Release Changes

- Organized the scientific entry points and two separate detector framework snapshots; original experimental folders are not modified.
- Preserved selected LSCD/LQE class bodies and YOLO13 modules; removed unused experimental backbone imports and replaced their broad YAML parser with a scoped selected-model parser.
- Added explicit detector dataset, pretrained checkpoint, output and scratch-test CLI options; final run paths follow the trainer's actual output directory.
- Made missing, duplicate, noninteger and stale augmentation split indices fail instead of silently using all discovered records.
- Added portable measured-calibration input/output options and analytical input validation; incomplete measurements remain excluded.
- Added dependency files, scientific interpretation notes, hardware instructions, tests, artificial smoke data, provenance, and a source-only archive audit.
- Explicitly select the legacy ONNX exporter when available, preserving the original export path on newer PyTorch versions.
- No retraining on the complete scientific dataset, revised accuracy claim, or new edge timing measurement is implied by this release.
