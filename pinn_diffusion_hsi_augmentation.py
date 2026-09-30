from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = ROOT / "ENVI_CORRECTED_CROPPED"
DEFAULT_OUTPUT_ROOT = ROOT / "ENVI_PINN_DIFFUSION_AUGMENTED"
DEFAULT_RUN_ROOT = ROOT / "pinn_diffusion_runs"
DEFAULT_SPLIT_JSON = None  # Default protocol uses separate DATA<class>_test folders.


CATEGORY_SPECS = [
    ("cmb", "DATACMB", "CMB"),
    ("dqb", "DATADQB", "DQB"),
    ("dwb", "DATADWB", "DWB"),
    ("wq", "DATAWQ", "WQ"),
    ("ymxb", "DATAYMXB", "YMXB"),
]


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
    interleave: str
    byte_order: int
    wavelengths: np.ndarray
    wavelength_units: str = "Nanometers"


@dataclass(frozen=True)
class SampleRecord:
    name: str
    prefix: str
    folder: str
    label: str
    class_index: int
    hdr_path: Path


@dataclass
class CubeNormalizer:
    low: float
    high: float

    @property
    def scale(self) -> float:
        return max(self.high - self.low, 1e-6)

    def transform(self, cube: np.ndarray) -> np.ndarray:
        x = (cube.astype(np.float32) - self.low) / self.scale
        return np.clip(x, 0.0, 1.0).astype(np.float32)

    def inverse(self, cube: np.ndarray) -> np.ndarray:
        x = cube.astype(np.float32) * self.scale + self.low
        x[~np.isfinite(x)] = self.low
        return np.maximum(x, 0.0).astype(np.float32)


@dataclass
class LatentScaler:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, latents: torch.Tensor) -> "LatentScaler":
        mean = latents.mean(dim=0, keepdim=True)
        std = latents.std(dim=0, keepdim=True).clamp_min(1e-5)
        return cls(mean=mean, std=std)

    def transform(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.mean.to(latents.device)) / self.std.to(latents.device)

    def inverse(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.std.to(latents.device) + self.mean.to(latents.device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def natural_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 10**9, path.stem)


def parse_int(text: str, field: str, default: int | None = None) -> int:
    match = re.search(rf"^{re.escape(field)}\s*=\s*(-?\d+)", text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        if default is None:
            raise ValueError(f"Missing ENVI field: {field}")
        return default
    return int(match.group(1))


def parse_str(text: str, field: str, default: str = "") -> str:
    match = re.search(rf"^{re.escape(field)}\s*=\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return default
    return match.group(1).strip().strip("{}").strip()


def parse_wavelengths(text: str, bands: int) -> np.ndarray:
    match = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return np.linspace(411.0, 779.0, bands, dtype=np.float32)
    values = [float(item) for item in re.findall(r"-?\d+(?:\.\d+)?", match.group(1))]
    if len(values) != bands:
        return np.linspace(411.0, 779.0, bands, dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def read_envi_meta(hdr_path: Path) -> EnviMeta:
    text = hdr_path.read_text(encoding="utf-8", errors="ignore")
    bands = parse_int(text, "bands")
    return EnviMeta(
        samples=parse_int(text, "samples"),
        lines=parse_int(text, "lines"),
        bands=bands,
        data_type=parse_int(text, "data type"),
        interleave=parse_str(text, "interleave", "bsq").lower(),
        byte_order=parse_int(text, "byte order", 0),
        wavelengths=parse_wavelengths(text, bands),
        wavelength_units=parse_str(text, "wavelength units", "Nanometers"),
    )


def envi_dtype(meta: EnviMeta) -> np.dtype:
    if meta.data_type not in DTYPE_MAP:
        raise ValueError(f"Unsupported ENVI data type: {meta.data_type}")
    dtype = np.dtype(DTYPE_MAP[meta.data_type])
    if meta.byte_order == 1:
        return dtype.newbyteorder(">")
    return dtype.newbyteorder("<")


def read_envi_cube(hdr_path: Path) -> tuple[EnviMeta, np.ndarray]:
    meta = read_envi_meta(hdr_path)
    raw_path = hdr_path.with_suffix(".raw")
    expected = meta.bands * meta.lines * meta.samples
    data = np.fromfile(raw_path, dtype=envi_dtype(meta), count=expected)
    if data.size != expected:
        raise ValueError(f"Raw size mismatch: {raw_path}, got {data.size}, expected {expected}")

    interleave = meta.interleave.lower()
    if interleave == "bsq":
        cube = data.reshape(meta.bands, meta.lines, meta.samples)
    elif interleave == "bil":
        cube = data.reshape(meta.lines, meta.bands, meta.samples).transpose(1, 0, 2)
    elif interleave == "bip":
        cube = data.reshape(meta.lines, meta.samples, meta.bands).transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported ENVI interleave: {meta.interleave}")

    cube = cube.astype(np.float32, copy=False)
    cube[~np.isfinite(cube)] = 0.0
    return meta, cube


def fit_cube_to_template(cube: np.ndarray, template: EnviMeta) -> np.ndarray:
    if cube.shape[0] != template.bands:
        raise ValueError(f"Band mismatch: cube has {cube.shape[0]}, template has {template.bands}")
    bands, src_h, src_w = cube.shape
    dst_h = template.lines
    dst_w = template.samples
    if src_h == dst_h and src_w == dst_w:
        return cube.astype(np.float32, copy=False)

    fitted = np.zeros((bands, dst_h, dst_w), dtype=np.float32)
    src_y0 = max((src_h - dst_h) // 2, 0)
    src_x0 = max((src_w - dst_w) // 2, 0)
    copy_h = min(src_h, dst_h)
    copy_w = min(src_w, dst_w)
    src_y1 = src_y0 + copy_h
    src_x1 = src_x0 + copy_w
    dst_y0 = max((dst_h - src_h) // 2, 0)
    dst_x0 = max((dst_w - src_w) // 2, 0)
    dst_y1 = dst_y0 + copy_h
    dst_x1 = dst_x0 + copy_w
    fitted[:, dst_y0:dst_y1, dst_x0:dst_x1] = cube[:, src_y0:src_y1, src_x0:src_x1]
    return fitted


def read_fitted_cube(hdr_path: Path, template: EnviMeta) -> tuple[EnviMeta, np.ndarray]:
    meta, cube = read_envi_cube(hdr_path)
    if meta.bands != template.bands:
        raise ValueError(f"Band mismatch at {hdr_path}: {meta.bands} != {template.bands}")
    if not np.allclose(meta.wavelengths, template.wavelengths):
        raise ValueError(f"Wavelength mismatch at {hdr_path}")
    return meta, fit_cube_to_template(cube, template)


def format_wavelengths(wavelengths: np.ndarray, per_line: int = 10) -> str:
    chunks = []
    values = [f"{float(v):.4f}" for v in wavelengths]
    for start in range(0, len(values), per_line):
        chunks.append("  " + ", ".join(values[start : start + per_line]))
    return "{\n" + ",\n".join(chunks) + "\n}"


def write_envi_cube(hdr_path: Path, cube: np.ndarray, template: EnviMeta, description: str) -> None:
    hdr_path.parent.mkdir(parents=True, exist_ok=True)
    cube = np.asarray(cube, dtype=np.float32)
    if cube.shape != (template.bands, template.lines, template.samples):
        raise ValueError(f"Cube shape {cube.shape} does not match template {(template.bands, template.lines, template.samples)}")
    cube[~np.isfinite(cube)] = 0.0
    cube.astype("<f4", copy=False).tofile(hdr_path.with_suffix(".raw"))

    header = "\n".join(
        [
            "ENVI",
            f"description = {{{description}}}",
            f"samples = {template.samples}",
            f"lines   = {template.lines}",
            f"bands   = {template.bands}",
            "header offset = 0",
            "file type = ENVI Standard",
            "interleave = bsq",
            "sensor type = Unknown",
            "byte order = 0",
            "data type = 4",
            f"wavelength units = {template.wavelength_units}",
            f"wavelength = {format_wavelengths(template.wavelengths)}",
            "",
        ]
    )
    hdr_path.write_text(header, encoding="utf-8")


def discover_records(data_root: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for class_index, (prefix, folder, label) in enumerate(CATEGORY_SPECS):
        folder_path = data_root / folder
        if not folder_path.exists():
            raise FileNotFoundError(f"Missing class folder: {folder_path}")
        hdrs = sorted(folder_path.glob("*.hdr"), key=natural_key)
        for hdr in hdrs:
            records.append(
                SampleRecord(
                    name=f"{prefix}_{hdr.stem}",
                    prefix=prefix,
                    folder=folder,
                    label=label,
                    class_index=class_index,
                    hdr_path=hdr,
                )
            )
    if not records:
        raise FileNotFoundError(f"No ENVI .hdr files found below {data_root}")
    return records


def choose_template_meta(records: list[SampleRecord]) -> EnviMeta:
    shape_counts: dict[tuple[int, int, int], int] = {}
    first_for_shape: dict[tuple[int, int, int], Path] = {}
    for rec in records:
        meta = read_envi_meta(rec.hdr_path)
        key = (meta.bands, meta.lines, meta.samples)
        shape_counts[key] = shape_counts.get(key, 0) + 1
        first_for_shape.setdefault(key, rec.hdr_path)
    shape = max(shape_counts.items(), key=lambda item: (item[1], item[0][1] * item[0][2]))[0]
    template = read_envi_meta(first_for_shape[shape])
    print(f"[template] using most common shape bands={shape[0]}, lines={shape[1]}, samples={shape[2]} ({shape_counts[shape]} cubes)")
    if len(shape_counts) > 1:
        shape_text = ", ".join(f"{key}:{value}" for key, value in sorted(shape_counts.items()))
        print(f"[template] shape distribution: {shape_text}")
    return template


def select_fit_records(records: list[SampleRecord], split_json: Path | None, fit_subset: str) -> list[SampleRecord]:
    if fit_subset == "all":
        return records
    if split_json is None:
        if fit_subset != "train":
            raise ValueError("Without --split-json, only --fit-subset train/all is valid.")
        print(f"[split] Using {len(records)} records from training folders only.")
        return records
    if not split_json.is_file():
        raise FileNotFoundError(f"Explicit split JSON does not exist: {split_json}")
    split = json.loads(split_json.read_text(encoding="utf-8"))
    indices = split.get(fit_subset)
    if indices is None:
        raise ValueError(f"Split JSON has no key '{fit_subset}': {split_json}")
    if not isinstance(indices, list) or any(type(i) is not int for i in indices):
        raise ValueError(f"Split '{fit_subset}' must be a JSON list of integer indices")
    indices_int = indices
    if not indices_int or len(set(indices_int)) != len(indices_int):
        raise ValueError(f"Split '{fit_subset}' must contain unique, nonempty indices")
    if min(indices_int) < 0 or max(indices_int) >= len(records):
        raise IndexError(
            f"Split '{fit_subset}' in {split_json} references record index {max(indices_int)}, "
            f"but only {len(records)} training records were discovered. "
            "Regenerate the split JSON for the current folder layout; do not bypass the split to suppress this error."
        )
    selected = [records[i] for i in indices_int]
    print(f"[split] Using {len(selected)} '{fit_subset}' records from {split_json}")
    return selected


def limit_records_per_class(records: list[SampleRecord], max_train_samples: int | None, seed: int) -> list[SampleRecord]:
    if max_train_samples is None or max_train_samples <= 0 or max_train_samples >= len(records):
        return records
    rng = random.Random(seed)
    chosen = records[:]
    rng.shuffle(chosen)
    chosen = sorted(chosen[:max_train_samples], key=lambda r: (r.class_index, natural_key(r.hdr_path)))
    print(f"[debug] Limited fitting set to {len(chosen)} records.")
    return chosen


def summarize_records(records: Iterable[SampleRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec.label] = counts.get(rec.label, 0) + 1
    return counts


def estimate_normalizer(
    records: list[SampleRecord],
    template: EnviMeta,
    low_percentile: float,
    high_percentile: float,
    values_per_cube: int,
    seed: int,
) -> CubeNormalizer:
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    for idx, rec in enumerate(records, start=1):
        _meta, cube = read_fitted_cube(rec.hdr_path, template)
        flat = cube.reshape(-1)
        if values_per_cube > 0 and flat.size > values_per_cube:
            positions = rng.integers(0, flat.size, size=values_per_cube)
            flat = flat[positions]
        samples.append(flat.astype(np.float32, copy=False))
        if idx % 50 == 0 or idx == len(records):
            print(f"[normalizer] sampled {idx}/{len(records)} cubes")
    values = np.concatenate(samples)
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    if high <= low:
        high = low + 1.0
    print(f"[normalizer] raw range mapped to [0,1]: p{low_percentile}={low:.6g}, p{high_percentile}={high:.6g}")
    return CubeNormalizer(low=low, high=high)


class EnviCubeDataset(Dataset):
    def __init__(self, records: list[SampleRecord], normalizer: CubeNormalizer, template: EnviMeta):
        self.records = records
        self.normalizer = normalizer
        self.template = template

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rec = self.records[index]
        _meta, cube = read_fitted_cube(rec.hdr_path, self.template)
        cube = self.normalizer.transform(cube)
        return torch.from_numpy(cube), torch.tensor(rec.class_index, dtype=torch.long), rec.name


def group_norm_channels(channels: int) -> int:
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(group_norm_channels(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(group_norm_channels(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CubeAutoencoder(nn.Module):
    def __init__(
        self,
        bands: int,
        height: int,
        width: int,
        num_classes: int,
        latent_dim: int,
        base_channels: int,
        label_dim: int = 32,
    ):
        super().__init__()
        self.bands = bands
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.encoder = nn.Sequential(
            ConvBlock(bands, c1, stride=2),
            ConvBlock(c1, c2, stride=2),
            ConvBlock(c2, c3, stride=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, bands, height, width)
            encoded = self.encoder(dummy)
        self.feature_shape = tuple(encoded.shape[1:])
        feature_size = int(np.prod(self.feature_shape))
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, latent_dim),
        )
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim + label_dim, feature_size),
            nn.SiLU(),
        )
        self.dec1 = ConvBlock(c3, c2)
        self.dec2 = ConvBlock(c2, c1)
        self.dec3 = ConvBlock(c1, c1)
        self.out = nn.Conv2d(c1, bands, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.encoder(x))

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.label_embedding(labels)
        h = self.from_latent(torch.cat([z, emb], dim=1))
        h = h.view(z.shape[0], *self.feature_shape)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.dec1(h)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.dec2(h)
        h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
        h = self.dec3(h)
        h = F.interpolate(h, size=(self.height, self.width), mode="bilinear", align_corners=False)
        return torch.sigmoid(self.out(h))

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z, labels), z


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if half == 0:
            return t.float().unsqueeze(1)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim: int, num_classes: int, hidden_dim: int = 512, time_dim: int = 128, label_dim: int = 64):
        super().__init__()
        self.time_embedding = SinusoidalEmbedding(time_dim)
        self.label_embedding = nn.Embedding(num_classes, label_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_dim + label_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        y_emb = self.label_embedding(labels)
        return self.net(torch.cat([z_t, t_emb, y_emb], dim=1))


def make_radial_bin_ids(height: int, width: int, radial_bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    max_radius = float(radius.max()) + 1e-6
    bin_ids = np.floor(radius / max_radius * radial_bins).astype(np.int64)
    bin_ids = np.clip(bin_ids, 0, radial_bins - 1)
    counts = np.bincount(bin_ids.reshape(-1), minlength=radial_bins).astype(np.float32)
    counts[counts == 0] = 1.0
    centers_px = (np.arange(radial_bins, dtype=np.float32) + 0.5) * max_radius / radial_bins
    return bin_ids, counts, centers_px


def radial_profile_np(cube: np.ndarray, bin_ids: np.ndarray, bin_counts: np.ndarray) -> np.ndarray:
    bands = cube.shape[0]
    radial_bins = len(bin_counts)
    flat_bins = bin_ids.reshape(-1)
    flat = cube.reshape(bands, -1)
    profiles = np.empty((bands, radial_bins), dtype=np.float32)
    for band in range(bands):
        profiles[band] = np.bincount(flat_bins, weights=flat[band], minlength=radial_bins)[:radial_bins] / bin_counts
    return profiles


def build_radial_masks(height: int, width: int, radial_bins: int, device: torch.device) -> torch.Tensor:
    bin_ids, counts, _centers = make_radial_bin_ids(height, width, radial_bins)
    masks = np.zeros((radial_bins, height, width), dtype=np.float32)
    for radius_bin in range(radial_bins):
        masks[radius_bin] = (bin_ids == radius_bin).astype(np.float32) / counts[radius_bin]
    return torch.from_numpy(masks).to(device)


@dataclass
class PhysicsReference:
    wavelengths_nm: torch.Tensor
    spectrum_mean: torch.Tensor
    spectrum_std: torch.Tensor
    energy_mean: torch.Tensor
    energy_std: torch.Tensor
    radial_mean: torch.Tensor
    radial_std: torch.Tensor
    radial_masks: torch.Tensor
    radial_centers_px: torch.Tensor

    def to(self, device: torch.device) -> "PhysicsReference":
        return PhysicsReference(
            wavelengths_nm=self.wavelengths_nm.to(device),
            spectrum_mean=self.spectrum_mean.to(device),
            spectrum_std=self.spectrum_std.to(device),
            energy_mean=self.energy_mean.to(device),
            energy_std=self.energy_std.to(device),
            radial_mean=self.radial_mean.to(device),
            radial_std=self.radial_std.to(device),
            radial_masks=self.radial_masks.to(device),
            radial_centers_px=self.radial_centers_px.to(device),
        )


def build_physics_reference(
    records: list[SampleRecord],
    normalizer: CubeNormalizer,
    template: EnviMeta,
    radial_bins: int,
    device: torch.device,
) -> PhysicsReference:
    num_classes = len(CATEGORY_SPECS)
    bands = template.bands
    bin_ids, bin_counts, centers_px = make_radial_bin_ids(template.lines, template.samples, radial_bins)
    spec_sum = np.zeros((num_classes, bands), dtype=np.float64)
    spec_sq = np.zeros((num_classes, bands), dtype=np.float64)
    energy_sum = np.zeros(num_classes, dtype=np.float64)
    energy_sq = np.zeros(num_classes, dtype=np.float64)
    radial_sum = np.zeros((num_classes, bands, radial_bins), dtype=np.float64)
    radial_sq = np.zeros((num_classes, bands, radial_bins), dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.float64)

    for idx, rec in enumerate(records, start=1):
        _meta, raw_cube = read_fitted_cube(rec.hdr_path, template)
        cube = normalizer.transform(raw_cube)
        spectrum = cube.mean(axis=(1, 2))
        energy = float(cube.mean())
        radial = radial_profile_np(cube, bin_ids, bin_counts)
        c = rec.class_index
        spec_sum[c] += spectrum
        spec_sq[c] += spectrum**2
        energy_sum[c] += energy
        energy_sq[c] += energy**2
        radial_sum[c] += radial
        radial_sq[c] += radial**2
        counts[c] += 1.0
        if idx % 50 == 0 or idx == len(records):
            print(f"[physics-ref] accumulated {idx}/{len(records)} cubes")

    counts_safe = np.maximum(counts, 1.0)
    spectrum_mean = spec_sum / counts_safe[:, None]
    spectrum_var = np.maximum(spec_sq / counts_safe[:, None] - spectrum_mean**2, 1e-6)
    energy_mean = energy_sum / counts_safe
    energy_var = np.maximum(energy_sq / counts_safe - energy_mean**2, 1e-6)
    radial_mean = radial_sum / counts_safe[:, None, None]
    radial_var = np.maximum(radial_sq / counts_safe[:, None, None] - radial_mean**2, 1e-6)

    radial_masks = build_radial_masks(template.lines, template.samples, radial_bins, device)
    return PhysicsReference(
        wavelengths_nm=torch.from_numpy(template.wavelengths.astype(np.float32)).to(device),
        spectrum_mean=torch.from_numpy(spectrum_mean.astype(np.float32)).to(device),
        spectrum_std=torch.from_numpy(np.sqrt(spectrum_var).astype(np.float32)).to(device),
        energy_mean=torch.from_numpy(energy_mean.astype(np.float32)).to(device),
        energy_std=torch.from_numpy(np.sqrt(energy_var).astype(np.float32)).to(device),
        radial_mean=torch.from_numpy(radial_mean.astype(np.float32)).to(device),
        radial_std=torch.from_numpy(np.sqrt(radial_var).astype(np.float32)).to(device),
        radial_masks=radial_masks,
        radial_centers_px=torch.from_numpy(centers_px.astype(np.float32)).to(device),
    )


def linear_resample_profiles(profiles: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    # profiles: B, D, R. scales: D. Values >1 sample farther out in each wavelength profile.
    bsz, bands, radial_bins = profiles.shape
    base = torch.arange(radial_bins, device=profiles.device, dtype=torch.float32)
    src = base.view(1, radial_bins) * scales.view(bands, 1)
    src = src.clamp(0.0, radial_bins - 1.0001)
    left = torch.floor(src).long()
    right = torch.clamp(left + 1, max=radial_bins - 1)
    weight = (src - left.float()).view(1, bands, radial_bins)
    left_values = torch.gather(profiles, 2, left.view(1, bands, radial_bins).expand(bsz, -1, -1))
    right_values = torch.gather(profiles, 2, right.view(1, bands, radial_bins).expand(bsz, -1, -1))
    return left_values * (1.0 - weight) + right_values * weight


class PhysicsLoss(nn.Module):
    """
    Intensity-only physics loss for near-field hyperspectral diffraction.

    The exact Fresnel forward model needs the complex optical field at the object plane.
    The current data contain CMOS intensities only, so this loss uses a weak Fresnel
    form that is still physically meaningful:

    1. Detector intensity is non-negative and bounded after percentile scaling.
    2. Mean spectra should be smooth over wavelength unless real class data support peaks.
    3. Neighboring CMOS pixels should not form isolated single-pixel spikes.
    4. Total energy and class mean spectra should remain in the empirical class envelope.
    5. Fresnel fringe radii scale approximately with sqrt(lambda * z) for fixed geometry,
       so radial profiles are compared after wavelength-dependent sqrt(lambda) rescaling.

    The class-envelope terms keep generated samples useful for classification, while
    the Fresnel radial-scaling term encodes the near-field geometry without pretending
    that unknown spore phase and refractive-index maps are available.
    """

    def __init__(
        self,
        reference: PhysicsReference,
        spectral_weight: float,
        spatial_weight: float,
        range_weight: float,
        energy_weight: float,
        class_spectrum_weight: float,
        radial_reference_weight: float,
        fresnel_scaling_weight: float,
        spectrum_tolerance: float,
        radial_tolerance: float,
    ):
        super().__init__()
        self.reference = reference
        self.spectral_weight = spectral_weight
        self.spatial_weight = spatial_weight
        self.range_weight = range_weight
        self.energy_weight = energy_weight
        self.class_spectrum_weight = class_spectrum_weight
        self.radial_reference_weight = radial_reference_weight
        self.fresnel_scaling_weight = fresnel_scaling_weight
        self.spectrum_tolerance = spectrum_tolerance
        self.radial_tolerance = radial_tolerance

    def radial_profiles(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bdhw,rhw->bdr", x, self.reference.radial_masks)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
        components: dict[str, torch.Tensor] = {}
        zero = x.new_tensor(0.0)

        if self.range_weight > 0:
            components["range"] = F.relu(-x).pow(2).mean() + F.relu(x - 1.0).pow(2).mean()
        else:
            components["range"] = zero

        if self.spectral_weight > 0 and x.shape[1] > 2:
            d2 = x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]
            components["spectral_smooth"] = d2.pow(2).mean()
        else:
            components["spectral_smooth"] = zero

        if self.spatial_weight > 0:
            dy = x[:, :, 1:, :] - x[:, :, :-1, :]
            dx = x[:, :, :, 1:] - x[:, :, :, :-1]
            components["spatial_tv"] = dx.pow(2).mean() + dy.pow(2).mean()
        else:
            components["spatial_tv"] = zero

        spectrum = x.mean(dim=(2, 3))
        if self.class_spectrum_weight > 0:
            mean = self.reference.spectrum_mean[labels]
            std = (self.reference.spectrum_std[labels] * self.spectrum_tolerance).clamp_min(0.03)
            components["class_spectrum"] = ((spectrum - mean) / std).pow(2).mean()
        else:
            components["class_spectrum"] = zero

        if self.energy_weight > 0:
            energy = x.mean(dim=(1, 2, 3))
            mean = self.reference.energy_mean[labels]
            std = (self.reference.energy_std[labels] * self.spectrum_tolerance).clamp_min(0.02)
            components["energy"] = ((energy - mean) / std).pow(2).mean()
        else:
            components["energy"] = zero

        needs_radial = self.radial_reference_weight > 0 or self.fresnel_scaling_weight > 0
        radial = self.radial_profiles(x) if needs_radial else None

        if self.radial_reference_weight > 0 and radial is not None:
            mean = self.reference.radial_mean[labels]
            std = (self.reference.radial_std[labels] * self.radial_tolerance).clamp_min(0.035)
            components["radial_reference"] = ((radial - mean) / std).pow(2).mean()
        else:
            components["radial_reference"] = zero

        if self.fresnel_scaling_weight > 0 and radial is not None and x.shape[1] > 2:
            wavelengths = self.reference.wavelengths_nm.to(x.device)
            ref_wavelength = torch.mean(wavelengths)
            scales = torch.sqrt((wavelengths / ref_wavelength).clamp_min(1e-6))
            aligned = linear_resample_profiles(radial, scales)
            d2 = aligned[:, 2:] - 2.0 * aligned[:, 1:-1] + aligned[:, :-2]
            components["fresnel_scaling"] = d2.pow(2).mean()
        else:
            components["fresnel_scaling"] = zero

        total = (
            self.range_weight * components["range"]
            + self.spectral_weight * components["spectral_smooth"]
            + self.spatial_weight * components["spatial_tv"]
            + self.energy_weight * components["energy"]
            + self.class_spectrum_weight * components["class_spectrum"]
            + self.radial_reference_weight * components["radial_reference"]
            + self.fresnel_scaling_weight * components["fresnel_scaling"]
        )
        components["total"] = total
        return components


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    posterior_variance: torch.Tensor


def make_diffusion_schedule(timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> DiffusionSchedule:
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]])
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars).clamp_min(1e-8)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
        posterior_variance=posterior_variance.clamp_min(1e-12),
    )


def extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.view(-1, *([1] * (target.ndim - 1)))


def train_autoencoder(
    model: CubeAutoencoder,
    loader: DataLoader,
    physics_loss: PhysicsLoss,
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_autoencoder, weight_decay=args.weight_decay)
    best = float("inf")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs_autoencoder + 1):
        model.train()
        sums: dict[str, float] = {}
        seen = 0
        for cubes, labels, _names in loader:
            cubes = cubes.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device)
            optimizer.zero_grad(set_to_none=True)
            recon, z = model(cubes, labels)
            recon_loss = F.mse_loss(recon, cubes)
            phys = physics_loss(recon, labels)
            latent_reg = z.pow(2).mean()
            loss = recon_loss + args.autoencoder_physics_weight * phys["total"] + args.latent_l2_weight * latent_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch = cubes.shape[0]
            seen += batch
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu()) * batch
            sums["recon"] = sums.get("recon", 0.0) + float(recon_loss.detach().cpu()) * batch
            sums["physics"] = sums.get("physics", 0.0) + float(phys["total"].detach().cpu()) * batch
        avg = {key: value / max(seen, 1) for key, value in sums.items()}
        print(
            f"[ae] epoch {epoch:03d}/{args.epochs_autoencoder} "
            f"loss={avg['loss']:.6f} recon={avg['recon']:.6f} physics={avg['physics']:.6f}"
        )
        if avg["loss"] < best:
            best = avg["loss"]
            torch.save({"model": model.state_dict(), "best_loss": best}, checkpoint_path)

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model"])


@torch.no_grad()
def encode_latents(model: CubeAutoencoder, loader: DataLoader, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    model.eval()
    latents: list[torch.Tensor] = []
    labels_out: list[torch.Tensor] = []
    names_out: list[str] = []
    for cubes, labels, names in loader:
        cubes = cubes.to(device=device, dtype=torch.float32)
        z = model.encode(cubes).cpu()
        latents.append(z)
        labels_out.append(labels.cpu())
        names_out.extend(list(names))
    return torch.cat(latents, dim=0), torch.cat(labels_out, dim=0), names_out


def train_diffusion(
    denoiser: LatentDenoiser,
    autoencoder: CubeAutoencoder,
    latents: torch.Tensor,
    labels: torch.Tensor,
    scaler: LatentScaler,
    physics_loss: PhysicsLoss,
    device: torch.device,
    args: argparse.Namespace,
    checkpoint_path: Path,
) -> None:
    z_scaled = scaler.transform(latents)
    dataset = TensorDataset(z_scaled.float(), labels.long())
    loader = DataLoader(dataset, batch_size=args.latent_batch_size, shuffle=True, drop_last=False)
    schedule = make_diffusion_schedule(args.timesteps, args.beta_start, args.beta_end, device)
    optimizer = torch.optim.AdamW(denoiser.parameters(), lr=args.lr_diffusion, weight_decay=args.weight_decay)
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad_(False)

    best = float("inf")
    global_step = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs_diffusion + 1):
        denoiser.train()
        sums: dict[str, float] = {}
        seen = 0
        for clean_z, y in loader:
            clean_z = clean_z.to(device=device, dtype=torch.float32)
            y = y.to(device=device)
            batch = clean_z.shape[0]
            t = torch.randint(0, args.timesteps, (batch,), device=device, dtype=torch.long)
            noise = torch.randn_like(clean_z)
            z_t = extract(schedule.sqrt_alpha_bars, t, clean_z) * clean_z + extract(
                schedule.sqrt_one_minus_alpha_bars, t, clean_z
            ) * noise
            pred_noise = denoiser(z_t, t, y)
            noise_loss = F.mse_loss(pred_noise, noise)
            loss = noise_loss
            physics_value = clean_z.new_tensor(0.0)

            if args.diffusion_physics_weight > 0 and global_step % args.diffusion_physics_every == 0:
                pred_x0 = (z_t - extract(schedule.sqrt_one_minus_alpha_bars, t, clean_z) * pred_noise) / extract(
                    schedule.sqrt_alpha_bars, t, clean_z
                ).clamp_min(1e-6)
                decoded = autoencoder.decode(scaler.inverse(pred_x0), y)
                phys = physics_loss(decoded, y)
                physics_value = phys["total"]
                loss = loss + args.diffusion_physics_weight * physics_value

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), args.grad_clip)
            optimizer.step()
            global_step += 1

            seen += batch
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach().cpu()) * batch
            sums["noise"] = sums.get("noise", 0.0) + float(noise_loss.detach().cpu()) * batch
            sums["physics"] = sums.get("physics", 0.0) + float(physics_value.detach().cpu()) * batch

        avg = {key: value / max(seen, 1) for key, value in sums.items()}
        print(
            f"[ddpm] epoch {epoch:03d}/{args.epochs_diffusion} "
            f"loss={avg['loss']:.6f} noise={avg['noise']:.6f} physics={avg['physics']:.6f}"
        )
        if avg["loss"] < best:
            best = avg["loss"]
            torch.save(
                {
                    "model": denoiser.state_dict(),
                    "best_loss": best,
                    "latent_mean": scaler.mean,
                    "latent_std": scaler.std,
                },
                checkpoint_path,
            )

    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        denoiser.load_state_dict(payload["model"])


@torch.no_grad()
def sample_ddpm(
    denoiser: LatentDenoiser,
    labels: torch.Tensor,
    latent_dim: int,
    timesteps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
) -> torch.Tensor:
    denoiser.eval()
    schedule = make_diffusion_schedule(timesteps, beta_start, beta_end, device)
    z = torch.randn(labels.shape[0], latent_dim, device=device)
    for step in reversed(range(timesteps)):
        t = torch.full((labels.shape[0],), step, device=device, dtype=torch.long)
        pred_noise = denoiser(z, t, labels)
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bars[step]
        beta = schedule.betas[step]
        mean = (1.0 / torch.sqrt(alpha)) * (z - beta / torch.sqrt((1.0 - alpha_bar).clamp_min(1e-8)) * pred_noise)
        if step > 0:
            z = mean + torch.sqrt(schedule.posterior_variance[step]) * torch.randn_like(z)
        else:
            z = mean
    return z


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def compute_class_spectra(
    records: list[SampleRecord],
    normalizer: CubeNormalizer,
    template: EnviMeta,
) -> dict[int, list[np.ndarray]]:
    spectra: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(CATEGORY_SPECS))}
    for rec in records:
        _meta, cube = read_fitted_cube(rec.hdr_path, template)
        x = normalizer.transform(cube)
        spectrum = x.mean(axis=(1, 2))
        if spectrum.shape[0] != template.bands:
            raise ValueError(f"Unexpected band count in {rec.hdr_path}")
        spectra[rec.class_index].append(spectrum.astype(np.float32))
    return spectra


@torch.no_grad()
def generate_synthetic_cubes(
    autoencoder: CubeAutoencoder,
    denoiser: LatentDenoiser,
    scaler: LatentScaler,
    normalizer: CubeNormalizer,
    template: EnviMeta,
    train_records: list[SampleRecord],
    output_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    autoencoder.eval()
    denoiser.eval()
    output_root.mkdir(parents=True, exist_ok=True)
    class_spectra = compute_class_spectra(train_records, normalizer, template)
    class_mean = {idx: np.mean(np.stack(items), axis=0) for idx, items in class_spectra.items() if items}
    manifest_path = output_root / "synthetic_manifest.csv"
    manifest_fields = [
        "sample",
        "label",
        "class_index",
        "hdr_path",
        "raw_path",
        "mean_intensity_norm",
        "std_intensity_norm",
        "spectrum_cosine_to_class_mean",
        "min_raw",
        "max_raw",
    ]
    rows: list[dict[str, object]] = []

    for class_index, (prefix, folder, label) in enumerate(CATEGORY_SPECS):
        remaining = args.per_class
        generated = 0
        out_dir = output_root / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        while remaining > 0:
            batch = min(args.generate_batch_size, remaining)
            labels = torch.full((batch,), class_index, device=device, dtype=torch.long)
            z_scaled = sample_ddpm(
                denoiser=denoiser,
                labels=labels,
                latent_dim=args.latent_dim,
                timesteps=args.timesteps,
                beta_start=args.beta_start,
                beta_end=args.beta_end,
                device=device,
            )
            z = scaler.inverse(z_scaled)
            cubes_norm = autoencoder.decode(z, labels).cpu().numpy()
            for item in range(batch):
                generated += 1
                sample_name = f"{prefix}_pinn_ddpm_{generated:04d}"
                hdr_path = out_dir / f"{sample_name}.hdr"
                cube_norm = np.clip(cubes_norm[item], 0.0, 1.0).astype(np.float32)
                cube_raw = normalizer.inverse(cube_norm)
                write_envi_cube(
                    hdr_path=hdr_path,
                    cube=cube_raw,
                    template=template,
                    description="Physics-informed latent DDPM synthetic hyperspectral diffraction cube",
                )
                spectrum = cube_norm.mean(axis=(1, 2))
                rows.append(
                    {
                        "sample": sample_name,
                        "label": label,
                        "class_index": class_index,
                        "hdr_path": str(hdr_path),
                        "raw_path": str(hdr_path.with_suffix(".raw")),
                        "mean_intensity_norm": float(cube_norm.mean()),
                        "std_intensity_norm": float(cube_norm.std()),
                        "spectrum_cosine_to_class_mean": cosine_similarity(spectrum, class_mean[class_index]),
                        "min_raw": float(cube_raw.min()),
                        "max_raw": float(cube_raw.max()),
                    }
                )
            remaining -= batch
            print(f"[generate] {label}: {generated}/{args.per_class}")

    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[generate] Wrote manifest: {manifest_path}")


def load_checkpoint_if_requested(
    autoencoder: CubeAutoencoder,
    denoiser: LatentDenoiser,
    ae_checkpoint: Path,
    ddpm_checkpoint: Path,
    device: torch.device,
) -> LatentScaler | None:
    scaler: LatentScaler | None = None
    if ae_checkpoint.exists():
        payload = torch.load(ae_checkpoint, map_location=device)
        autoencoder.load_state_dict(payload["model"])
        print(f"[checkpoint] Loaded autoencoder: {ae_checkpoint}")
    if ddpm_checkpoint.exists():
        payload = torch.load(ddpm_checkpoint, map_location=device)
        denoiser.load_state_dict(payload["model"])
        scaler = LatentScaler(mean=payload["latent_mean"], std=payload["latent_std"])
        print(f"[checkpoint] Loaded diffusion model: {ddpm_checkpoint}")
    return scaler


def geometry_report(meta: EnviMeta, args: argparse.Namespace) -> dict[str, float]:
    wavelengths_m = meta.wavelengths.astype(np.float64) * 1e-9
    z_m = args.object_sensor_mm * 1e-3
    pixel_pitch_m = args.pixel_pitch_um * 1e-6
    fresnel_length_px = np.sqrt(wavelengths_m * z_m) / pixel_pitch_m
    crop_width_um = meta.samples * args.pixel_pitch_um
    crop_height_um = meta.lines * args.pixel_pitch_um
    object_radius_values_um = np.asarray([5.0, 25.0, 50.0], dtype=np.float64)
    mid_wavelength_m = float(np.mean(wavelengths_m))
    fresnel_numbers = (object_radius_values_um * 1e-6) ** 2 / (mid_wavelength_m * z_m)
    return {
        "pixel_pitch_um": args.pixel_pitch_um,
        "object_sensor_mm": args.object_sensor_mm,
        "crop_width_um": crop_width_um,
        "crop_height_um": crop_height_um,
        "fresnel_length_px_min": float(fresnel_length_px.min()),
        "fresnel_length_px_max": float(fresnel_length_px.max()),
        "fresnel_number_radius_5um": float(fresnel_numbers[0]),
        "fresnel_number_radius_25um": float(fresnel_numbers[1]),
        "fresnel_number_radius_50um": float(fresnel_numbers[2]),
    }


def print_dry_run_report(
    records: list[SampleRecord],
    fit_records: list[SampleRecord],
    template: EnviMeta,
    normalizer: CubeNormalizer,
    physics_loss: PhysicsLoss,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    print("\n[dry-run] Dataset summary")
    print(f"  all records: {len(records)} {summarize_records(records)}")
    print(f"  fit records: {len(fit_records)} {summarize_records(fit_records)}")
    print(f"  cube shape: bands={template.bands}, lines={template.lines}, samples={template.samples}")
    print(f"  wavelengths: {template.wavelengths[0]:.1f}-{template.wavelengths[-1]:.1f} nm")
    print(f"  normalizer: low={normalizer.low:.6g}, high={normalizer.high:.6g}")
    print("\n[dry-run] Geometry")
    for key, value in geometry_report(template, args).items():
        print(f"  {key}: {value:.6g}")
    cubes, labels, names = next(iter(loader))
    cubes = cubes.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device)
    with torch.no_grad():
        components = physics_loss(cubes, labels)
    print("\n[dry-run] Physics loss on real batch")
    print(f"  samples: {', '.join(list(names)[:5])}")
    for key, value in components.items():
        print(f"  {key}: {float(value.detach().cpu()):.6g}")
    print("\n[dry-run] OK. Remove --dry-run to train and generate synthetic ENVI cubes.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Physics-informed latent DDPM augmentation for ENVI hyperspectral diffraction cubes."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--split-json", type=Path, default=DEFAULT_SPLIT_JSON)
    parser.add_argument("--fit-subset", choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Debug limit; 0 means no limit.")
    parser.add_argument("--per-class", type=int, default=100, help="Synthetic cubes to generate for each class.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse checkpoints if present.")
    parser.add_argument("--no-physics", action="store_true", help="Disable all physics losses for a plain latent DDPM baseline.")

    parser.add_argument("--low-percentile", type=float, default=0.5)
    parser.add_argument("--high-percentile", type=float, default=99.5)
    parser.add_argument("--values-per-cube", type=int, default=20000)

    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--latent-batch-size", type=int, default=32)
    parser.add_argument("--generate-batch-size", type=int, default=4)
    parser.add_argument("--epochs-autoencoder", type=int, default=60)
    parser.add_argument("--epochs-diffusion", type=int, default=300)
    parser.add_argument("--lr-autoencoder", type=float, default=2e-4)
    parser.add_argument("--lr-diffusion", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--latent-l2-weight", type=float, default=1e-4)

    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)

    parser.add_argument("--autoencoder-physics-weight", type=float, default=0.1)
    parser.add_argument("--diffusion-physics-weight", type=float, default=0.02)
    parser.add_argument("--diffusion-physics-every", type=int, default=4)
    parser.add_argument("--radial-bins", type=int, default=48)
    parser.add_argument("--range-weight", type=float, default=0.05)
    parser.add_argument("--spectral-weight", type=float, default=0.08)
    parser.add_argument("--spatial-weight", type=float, default=0.02)
    parser.add_argument("--energy-weight", type=float, default=0.04)
    parser.add_argument("--class-spectrum-weight", type=float, default=0.04)
    parser.add_argument("--radial-reference-weight", type=float, default=0.03)
    parser.add_argument("--fresnel-scaling-weight", type=float, default=0.08)
    parser.add_argument("--spectrum-tolerance", type=float, default=2.5)
    parser.add_argument("--radial-tolerance", type=float, default=3.0)

    parser.add_argument("--pixel-pitch-um", type=float, default=1.4, help="OV5640 pixel pitch used for geometry reporting.")
    parser.add_argument("--source-pinhole-cm", type=float, default=60.0)
    parser.add_argument("--pinhole-object-cm", type=float, default=60.0)
    parser.add_argument("--object-sensor-mm", type=float, default=5.0)
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_run_config(args: argparse.Namespace, path: Path, normalizer: CubeNormalizer, template: EnviMeta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    payload["normalizer"] = asdict(normalizer)
    payload["template"] = {
        "samples": template.samples,
        "lines": template.lines,
        "bands": template.bands,
        "wavelength_min_nm": float(template.wavelengths.min()),
        "wavelength_max_nm": float(template.wavelengths.max()),
    }
    payload["geometry_report"] = geometry_report(template, args)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.no_physics:
        args.autoencoder_physics_weight = 0.0
        args.diffusion_physics_weight = 0.0
        args.range_weight = 0.0
        args.spectral_weight = 0.0
        args.spatial_weight = 0.0
        args.energy_weight = 0.0
        args.class_spectrum_weight = 0.0
        args.radial_reference_weight = 0.0
        args.fresnel_scaling_weight = 0.0
        print("[physics] Disabled all physics losses for plain latent DDPM baseline.")
    seed_everything(args.seed)
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    print(f"[device] {device}")

    records = discover_records(args.data_root)
    fit_records = select_fit_records(records, args.split_json, args.fit_subset)
    fit_records = limit_records_per_class(
        fit_records,
        max_train_samples=(args.max_train_samples if args.max_train_samples > 0 else None),
        seed=args.seed,
    )
    template = choose_template_meta(records)
    for rec in fit_records:
        meta = read_envi_meta(rec.hdr_path)
        if meta.bands != template.bands:
            raise ValueError(f"Band mismatch at {rec.hdr_path}: {meta.bands} != {template.bands}")
        if not np.allclose(meta.wavelengths, template.wavelengths):
            raise ValueError(f"Wavelength mismatch at {rec.hdr_path}")

    print(f"[data] all records: {len(records)} {summarize_records(records)}")
    print(f"[data] fit records: {len(fit_records)} {summarize_records(fit_records)}")
    print(f"[data] template cube: {template.bands} x {template.lines} x {template.samples}")

    normalizer = estimate_normalizer(
        fit_records,
        template=template,
        low_percentile=args.low_percentile,
        high_percentile=args.high_percentile,
        values_per_cube=args.values_per_cube,
        seed=args.seed,
    )
    reference = build_physics_reference(
        records=fit_records,
        normalizer=normalizer,
        template=template,
        radial_bins=args.radial_bins,
        device=device,
    )
    physics_loss = PhysicsLoss(
        reference=reference,
        spectral_weight=args.spectral_weight,
        spatial_weight=args.spatial_weight,
        range_weight=args.range_weight,
        energy_weight=args.energy_weight,
        class_spectrum_weight=args.class_spectrum_weight,
        radial_reference_weight=args.radial_reference_weight,
        fresnel_scaling_weight=args.fresnel_scaling_weight,
        spectrum_tolerance=args.spectrum_tolerance,
        radial_tolerance=args.radial_tolerance,
    )

    dataset = EnviCubeDataset(fit_records, normalizer, template)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    if args.dry_run:
        print_dry_run_report(records, fit_records, template, normalizer, physics_loss, loader, device, args)
        return

    run_root = args.run_root
    ckpt_root = run_root / "checkpoints"
    ae_checkpoint = ckpt_root / "cube_autoencoder.pt"
    ddpm_checkpoint = ckpt_root / "latent_ddpm.pt"
    save_run_config(args, run_root / "run_config.json", normalizer, template)

    autoencoder = CubeAutoencoder(
        bands=template.bands,
        height=template.lines,
        width=template.samples,
        num_classes=len(CATEGORY_SPECS),
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
    ).to(device)
    denoiser = LatentDenoiser(
        latent_dim=args.latent_dim,
        num_classes=len(CATEGORY_SPECS),
        hidden_dim=max(256, args.latent_dim * 4),
    ).to(device)

    scaler = None
    if args.resume:
        scaler = load_checkpoint_if_requested(autoencoder, denoiser, ae_checkpoint, ddpm_checkpoint, device)

    if not args.resume or not ae_checkpoint.exists():
        train_autoencoder(autoencoder, loader, physics_loss, device, args, ae_checkpoint)

    latents, labels, _names = encode_latents(autoencoder, loader, device)
    if scaler is None:
        scaler = LatentScaler.fit(latents)

    if not args.resume or not ddpm_checkpoint.exists():
        train_diffusion(denoiser, autoencoder, latents, labels, scaler, physics_loss, device, args, ddpm_checkpoint)

    generate_synthetic_cubes(
        autoencoder=autoencoder,
        denoiser=denoiser,
        scaler=scaler,
        normalizer=normalizer,
        template=template,
        train_records=fit_records,
        output_root=args.output_root,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()
