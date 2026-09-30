from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WAVELENGTH_START_NM = 400.0
WAVELENGTH_END_NM = 800.0

CATEGORIES = {
    "DQB": {
        "source_prefix": "DQB-00-",
        "output_dir": "DATADQB",
        "output_prefix": "dqb",
        "max_outputs": 1,
    },
    "WQ": {
        "source_prefix": "WQ10-00-",
        "output_dir": "DATAWQ",
        "output_prefix": "wq",
        "max_outputs": 1,
    },
    "YMXB": {
        "source_prefix": "YMXB-00-",
        "output_dir": "DATAYMXB",
        "output_prefix": "ymxb",
        "max_outputs": 110,
    },
}

RENAME_OUTPUTS = {
    "DATACMB": "cmb",
    "DATADWB": "dwb",
    "DATADQB": "dqb",
    "DATAWQ": "wq",
    "DATAYMXB": "ymxb",
}


def trailing_int(text: str) -> int:
    match = re.search(r"(\d+)$", text)
    if not match:
        raise ValueError(f"No trailing number found in {text!r}")
    return int(match.group(1))


def image_sort_key(path: Path) -> tuple[float, str]:
    try:
        return (float(path.stem), path.name)
    except ValueError:
        return (float("inf"), path.name)


def find_image_files(roi_dir: Path) -> list[Path]:
    files_by_name: dict[str, Path] = {}
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        for path in roi_dir.glob(pattern):
            files_by_name[path.name.lower()] = path
        for path in roi_dir.glob(pattern.upper()):
            files_by_name[path.name.lower()] = path
    return sorted(files_by_name.values(), key=image_sort_key)


def envi_header(width: int, height: int, bands: int, dtype: np.dtype) -> str:
    dtype_map = {
        "uint8": 1,
        "int16": 2,
        "int32": 3,
        "float32": 4,
        "float64": 5,
        "complex64": 6,
        "complex128": 9,
        "uint16": 12,
        "uint32": 13,
        "int64": 14,
        "uint64": 15,
    }
    envi_dtype = dtype_map.get(str(dtype))
    if envi_dtype is None:
        raise ValueError(f"Unsupported dtype for ENVI: {dtype}")

    byte_order = 0 if sys.byteorder == "little" else 1
    wavelengths = np.linspace(WAVELENGTH_START_NM, WAVELENGTH_END_NM, bands)
    wavelength_text = ", ".join(f"{wl:.4f}" for wl in wavelengths)

    return "\n".join(
        [
            "ENVI",
            (
                "description = {File created by Python script. "
                f"Wavelengths from {WAVELENGTH_START_NM} to {WAVELENGTH_END_NM} nm.}}"
            ),
            f"samples = {width}",
            f"lines   = {height}",
            f"bands   = {bands}",
            "header offset = 0",
            "file type = ENVI Standard",
            "interleave = bsq",
            "sensor type = Unknown",
            f"byte order = {byte_order}",
            f"data type = {envi_dtype}",
            "wavelength units = Nanometers",
            f"wavelength = {{{wavelength_text}}}",
        ]
    )


def convert_roi_to_envi(roi_dir: Path, output_base: Path) -> None:
    image_files = find_image_files(roi_dir)
    if not image_files:
        raise ValueError(f"No image files found in {roi_dir}")

    arrays: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for image_file in image_files:
        with Image.open(image_file) as image:
            array = np.asarray(image.convert("L"))
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(
                f"Shape mismatch in {roi_dir}: {image_file.name} has {array.shape}, expected {shape}"
            )
        arrays.append(array)

    cube = np.stack(arrays, axis=0)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    cube.tofile(output_base.with_suffix(".raw"))

    height, width = cube.shape[1], cube.shape[2]
    output_base.with_suffix(".hdr").write_text(
        envi_header(width=width, height=height, bands=cube.shape[0], dtype=cube.dtype),
        encoding="utf-8",
    )


def convert_category(name: str, config: dict[str, str]) -> int:
    sample_dirs = sorted(
        [
            child
            for child in ROOT.iterdir()
            if child.is_dir() and child.name.startswith(config["source_prefix"])
        ],
        key=lambda path: trailing_int(path.name),
    )
    output_dir = ROOT / config["output_dir"]
    counter = 1
    max_outputs = int(config.get("max_outputs", 0))

    print(f"\n{name}: {len(sample_dirs)} sample folders -> {output_dir.name}")
    for sample_dir in sample_dirs:
        roi_dirs = sorted(
            [
                child
                for child in sample_dir.iterdir()
                if child.is_dir() and child.name.startswith("ROI_")
            ],
            key=lambda path: trailing_int(path.name),
        )
        for roi_dir in roi_dirs:
            if max_outputs and counter > max_outputs:
                break
            output_base = output_dir / f"{config['output_prefix']}{counter}"
            convert_roi_to_envi(roi_dir, output_base)
            print(f"  {roi_dir.relative_to(ROOT)} -> {output_base.relative_to(ROOT)}")
            counter += 1
        if max_outputs and counter > max_outputs:
            break

    print(f"{name}: converted {counter - 1} hyperspectral groups")
    return counter - 1


def pair_sort_key(path: Path, prefix: str) -> tuple[int, str]:
    stem = path.stem
    if stem.lower().startswith(prefix.lower()):
        suffix = stem[len(prefix) :]
        if suffix.isdigit():
            return (int(suffix), stem)
    matches = re.findall(r"(\d+)", stem)
    if matches:
        return (int(matches[-1]), stem)
    return (10**9, stem)


def is_already_normalized(output_dir: Path, prefix: str) -> bool:
    hdr_files = sorted(output_dir.glob("*.hdr"), key=lambda p: pair_sort_key(p, prefix))
    if not hdr_files:
        return True

    expected = {f"{prefix}{index}" for index in range(1, len(hdr_files) + 1)}
    actual = {path.stem for path in hdr_files}
    if actual != expected:
        return False
    return all((output_dir / f"{stem}.raw").exists() for stem in expected)


def normalize_output_names(output_dir: Path, prefix: str) -> int:
    if is_already_normalized(output_dir, prefix):
        return len(list(output_dir.glob("*.hdr")))

    hdr_files = sorted(output_dir.glob("*.hdr"), key=lambda p: pair_sort_key(p, prefix))
    pairs: list[tuple[Path, Path]] = []
    for hdr_file in hdr_files:
        raw_file = hdr_file.with_suffix(".raw")
        if not raw_file.exists():
            raise FileNotFoundError(f"Missing raw pair for {hdr_file}")
        pairs.append((hdr_file, raw_file))

    temp_pairs: list[tuple[Path, Path, Path, Path]] = []
    for index, (hdr_file, raw_file) in enumerate(pairs, start=1):
        temp_hdr = output_dir / f"__renaming_{index}.hdr"
        temp_raw = output_dir / f"__renaming_{index}.raw"
        hdr_file.rename(temp_hdr)
        raw_file.rename(temp_raw)
        temp_pairs.append((temp_hdr, temp_raw, output_dir / f"{prefix}{index}.hdr", output_dir / f"{prefix}{index}.raw"))

    for temp_hdr, temp_raw, final_hdr, final_raw in temp_pairs:
        temp_hdr.rename(final_hdr)
        temp_raw.rename(final_raw)

    return len(pairs)


def normalize_all_outputs() -> None:
    print("\nNormalizing DATA* output names")
    for folder_name, prefix in RENAME_OUTPUTS.items():
        output_dir = ROOT / folder_name
        if not output_dir.exists():
            continue
        count = normalize_output_names(output_dir, prefix)
        print(f"  {folder_name}: {count} pairs -> {prefix}1..{prefix}{count}")


def main() -> None:
    normalize_only = "--normalize-only" in sys.argv
    if not normalize_only:
        for name, config in CATEGORIES.items():
            convert_category(name, config)
    normalize_all_outputs()


if __name__ == "__main__":
    main()
