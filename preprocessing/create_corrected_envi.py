from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SHIFT_REPORT = ROOT / "SPECTRA_CSV" / "band_shift_report.csv"
OUTPUT_ROOT = ROOT / "ENVI_CORRECTED_CROPPED"
COMMON_WAVELENGTHS = np.arange(400.0, 801.0, 1.0)

CATEGORIES = {
    "cmb": ROOT / "DATACMB",
    "dwb": ROOT / "DATADWB",
    "dqb": ROOT / "DATADQB",
    "wq": ROOT / "DATAWQ",
    "ymxb": ROOT / "DATAYMXB",
}

DTYPE_MAP = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


@dataclass(frozen=True)
class EnviMeta:
    samples: int
    lines: int
    bands: int
    data_type: int
    wavelengths: np.ndarray


def parse_int_field(text: str, field: str) -> int:
    match = re.search(rf"^{re.escape(field)}\s*=\s*(\d+)", text, flags=re.MULTILINE | re.IGNORECASE)
    if not match:
        raise ValueError(f"Missing ENVI header field: {field}")
    return int(match.group(1))


def parse_wavelengths(text: str, bands: int) -> np.ndarray:
    match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return np.linspace(400.0, 800.0, bands)
    values = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
    if len(values) != bands:
        return np.linspace(400.0, 800.0, bands)
    return np.asarray(values, dtype=np.float64)


def read_meta(hdr_path: Path) -> EnviMeta:
    text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    bands = parse_int_field(text, "bands")
    return EnviMeta(
        samples=parse_int_field(text, "samples"),
        lines=parse_int_field(text, "lines"),
        bands=bands,
        data_type=parse_int_field(text, "data type"),
        wavelengths=parse_wavelengths(text, bands),
    )


def load_cube(hdr_path: Path) -> tuple[EnviMeta, np.ndarray]:
    meta = read_meta(hdr_path)
    dtype = DTYPE_MAP.get(meta.data_type)
    if dtype is None:
        raise ValueError(f"Unsupported ENVI data type {meta.data_type}: {hdr_path}")

    raw_path = hdr_path.with_suffix(".raw")
    expected = meta.bands * meta.lines * meta.samples
    data = np.fromfile(raw_path, dtype=dtype, count=expected)
    if data.size != expected:
        raise ValueError(f"Raw size mismatch for {raw_path}: {data.size} values, expected {expected}")
    cube = data.reshape((meta.bands, meta.lines, meta.samples)).astype(np.float32)
    return meta, cube


def resample_cube_to_common(meta: EnviMeta, cube: np.ndarray) -> np.ndarray:
    if len(meta.wavelengths) == len(COMMON_WAVELENGTHS) and np.allclose(meta.wavelengths, COMMON_WAVELENGTHS):
        return cube

    positions = np.interp(COMMON_WAVELENGTHS, meta.wavelengths, np.arange(meta.bands, dtype=np.float64))
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    lower = np.clip(lower, 0, meta.bands - 1)
    upper = np.clip(upper, 0, meta.bands - 1)
    weights = (positions - lower).astype(np.float32)[:, None, None]
    return cube[lower] * (1.0 - weights) + cube[upper] * weights


def apply_shift(cube: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.full_like(cube, np.nan, dtype=np.float32)
    if shift > 0:
        shifted[shift:] = cube[:-shift]
    elif shift < 0:
        shifted[:shift] = cube[-shift:]
    else:
        shifted[:] = cube
    return shifted


def read_shifts() -> dict[str, dict[str, int]]:
    shifts: dict[str, dict[str, int]] = {}
    with SHIFT_REPORT.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            shifts.setdefault(row["category"], {})[row["sample"]] = int(row["shift_bands"])
    return shifts


def envi_header(width: int, height: int, wavelengths: np.ndarray) -> str:
    wavelength_text = ", ".join(f"{value:.4f}" for value in wavelengths)
    return "\n".join(
        [
            "ENVI",
            "description = {Corrected and cropped ENVI cube created from wavelength-shift report.}",
            f"samples = {width}",
            f"lines   = {height}",
            f"bands   = {len(wavelengths)}",
            "header offset = 0",
            "file type = ENVI Standard",
            "interleave = bsq",
            "sensor type = Unknown",
            "byte order = 0",
            "data type = 4",
            "wavelength units = Nanometers",
            f"wavelength = {{{wavelength_text}}}",
        ]
    )


def common_valid_mask(all_shifts: list[int]) -> np.ndarray:
    valid = np.ones(len(COMMON_WAVELENGTHS), dtype=bool)
    for shift in all_shifts:
        current = np.zeros(len(COMMON_WAVELENGTHS), dtype=bool)
        if shift > 0:
            current[shift:] = True
        elif shift < 0:
            current[:shift] = True
        else:
            current[:] = True
        valid &= current
    return valid


def main() -> None:
    shifts = read_shifts()
    all_shifts = [shift for category_shifts in shifts.values() for shift in category_shifts.values()]
    valid = common_valid_mask(all_shifts)
    cropped_wavelengths = COMMON_WAVELENGTHS[valid]

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUTPUT_ROOT}")
    print(
        "Cropped wavelength range: "
        f"{cropped_wavelengths[0]:.0f}-{cropped_wavelengths[-1]:.0f} nm "
        f"({len(cropped_wavelengths)} bands)"
    )

    for category, input_dir in CATEGORIES.items():
        output_dir = OUTPUT_ROOT / input_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        category_shifts = shifts.get(category, {})
        hdr_files = sorted(input_dir.glob("*.hdr"), key=lambda p: int(re.search(r"(\d+)$", p.stem).group(1)))
        hdr_files = hdr_files[:100]

        for index, hdr_path in enumerate(hdr_files, start=1):
            if hdr_path.stem not in category_shifts:
                raise ValueError(f"Missing shift for {category}/{hdr_path.stem}")

            meta, cube = load_cube(hdr_path)
            common_cube = resample_cube_to_common(meta, cube)
            shifted_cube = apply_shift(common_cube, category_shifts[hdr_path.stem])
            cropped_cube = shifted_cube[valid].astype(np.float32)

            output_base = output_dir / hdr_path.stem
            cropped_cube.tofile(output_base.with_suffix(".raw"))
            output_base.with_suffix(".hdr").write_text(
                envi_header(width=meta.samples, height=meta.lines, wavelengths=cropped_wavelengths),
                encoding="utf-8",
            )

            if index % 20 == 0 or index == len(hdr_files):
                print(f"{category}: {index}/{len(hdr_files)}")


if __name__ == "__main__":
    main()
