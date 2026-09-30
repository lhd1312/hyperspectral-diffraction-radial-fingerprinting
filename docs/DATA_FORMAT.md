# Data Formats and Splits

## Labels

| Internal label | Manuscript label | Material |
| --- | --- | --- |
| CMB | Fg | Fusarium head blight conidia |
| DQB | Uv | Rice false smut chlamydospores |
| DWB | Po | Rice blast conidia |
| WQ | PVC | 10 micrometre PVC microspheres |
| YMXB | Pp | Southern corn rust spores |

Internal labels remain unchanged for checkpoint compatibility. Confirm isolate/species details against the experimental records, not filename abbreviations.

## Corrected ROI Experiment

```text
ENVI_CORRECTED_CROPPED/
  DATACMB/       # real training .hdr + .raw pairs
  DATACMB_test/  # real held-out pairs
  DATADQB/
  DATADQB_test/
  DATADWB/
  DATADWB_test/
  DATAWQ/
  DATAWQ_test/
  DATAYMXB/
  DATAYMXB_test/
```

The study configuration has 70 training and 30 test ROIs per class, 100 x 100 spatial pixels and 369 wavelengths (411 through 779 nm). Python arrays are band-first `[369,100,100]`. The reader handles BSQ/BIL/BIP; generation exports little-endian float32 BSQ with matching header and wavelengths. Header offset is assumed zero and the raw filename shares the header stem. Supply valid wavelength metadata; the historical reader otherwise falls back to 411-779 nm. Do not apply that fallback to a different acquisition.

Archived generator settings and the deployment classifier's selected wavelengths are supplied in `experiment_configs/`. The wavelength CSV contains configuration only, without experimental feature-ranking scores.

Augmentation discovery reads only `DATA<class>`, not `_test`. An explicit index split is relative to the current class/numeric filename ordering; stale, duplicate, missing, or out-of-range lists fail. Prefer immutable filename manifests for auditing. Splitting ROIs alone does not prove independence between preparations or slides: include those identifiers in the data release and evaluate grouping.

The `preprocessing/` scripts preserve historical conversion/alignment procedures. Their source-prefix tables and category-specific spectral shifts are experiment-specific; they are **not** an inference procedure for an unknown class. The deployment classifier is trained separately without class-dependent alignment. Treat conversion/renaming tools as batch operations and work on a copy.

## Class-Independent Deployment Training

`train_class_independent_deployment_classifier.py` expects `DATACMB`, `DATADQB`, etc. directly beneath `--data-root`, containing the uncorrected original ROI cubes. It numerically sorts filenames and reserves the first 70/last 30 per class by default. Do not pass the separated corrected ROI tree above and assume it has the same split semantics.

## Full-Field Sequence and Detection

```text
oridata/
  CMB_1/
    pos_-896000.jpg
    ...
    pos_700000.jpg
```

A full sequence contains 400 grayscale frames, with stage-position increments of 4000. The study maps the first and last positions to 400 and 800 nm. Thus the endpoint-inclusive sampling interval is `400/399 nm`, not exactly 1 nm. All frames must have the same shape and field of view; a filename alone cannot establish optical wavelength calibration.

The pseudo-RGB image combines the chosen single-band image, a PCA component, and fringe-energy information. It is a detector input, not a physical color rendering. Annotate once per registered field, not independently at all 400 wavelengths.

Pose data layout:

```text
data/pose/
  images/train/*.png
  images/val/*.png
  labels/train/*.txt
  labels/val/*.txt
```

Each label line: `0 cx cy width height kx ky visibility`, normalized by the original image width/height; visibility is `2` for a visible annotated center. There is one detector class (`particle`) and one keypoint. Boxes are 100 x 100 native pixels around the diffraction center. Reject centers for which a complete square crop lies outside the field. Review overlapping objects individually only when their centers are distinguishable.

Create an absolute dataset YAML for the local checkout:

```bash
python tools/make_pose_yaml.py --root data/pose --output data/pose/data.yaml
```

Do not put different wavelengths or crops from the same field into different detector splits. Preserve annotation provenance, original field IDs, and the dataset version.

## Measured Quantification

Use `examples/calibration_template.csv` as a header schema, not as evidence. One row represents one field; five measured fields make one preparation. `preparation_id` distinguishes independent preparations, and `field_id` distinguishes fields within a preparation. `reference_concentration_particles_ml` must be calculated from the actual counting-chamber geometry and dilution factor.

The archived analysis includes only complete five-field reference/count preparations; it does not fill in missing observations. It recognizes the study's included gradients C0, C1, C3, C5, C7, C8 and relabels them G0-G5 for figures. The independent regression unit is a preparation, not an individual field. Blank measurements and new materials require their own validation protocol.
