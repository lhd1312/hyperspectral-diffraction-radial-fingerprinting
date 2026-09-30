"""Audit and optionally archive source only, excluding datasets and runtime artifacts."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {".github", "deployment", "docs", "examples", "experiment_configs", "models",
               "preprocessing", "tests", "tools", "vendor"}
EXCLUDE = {".git", ".cache", ".smoke", ".pytest_cache", ".venv", "__pycache__",
           "results", "outputs", "runs", "captures", "pydeps", "artifacts", "shili"}
EXTENSIONS = {".py", ".md", ".txt", ".json", ".csv", ".yaml", ".yml", ".toml", ".ini", ".sh"}
SPECIAL_NAMES = {"LICENSE", ".gitignore", ".gitattributes"}
PRIVATE_PATTERNS = [
    ("private device IP", re.compile(r"\b192\.168\.\d{1,3}\.\d{1,3}\b")),
    ("personal Windows path", re.compile(r"[A-Za-z]:[\\/](?:Users|2026duobing)[\\/]", re.I)),
    ("private key", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")),
    ("GitHub credential", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,})\b")),
]


def source_files():
    for directory, dirs, files in os.walk(ROOT, topdown=True):
        base = Path(directory)
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE and
                         (base != ROOT or d in SOURCE_DIRS))
        for name in sorted(files):
            path = base / name
            if path.suffix not in EXTENSIONS and name not in SPECIAL_NAMES:
                continue
            if path.is_symlink():
                raise ValueError(f"Symlinks are not included: {path.relative_to(ROOT)}")
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--normalize-lines", action="store_true", help="Normalize curated source text to LF")
    parser.add_argument("--archive", type=Path, help="Create a new source-only ZIP outside the source tree")
    args = parser.parse_args()
    paths = list(source_files())
    if args.normalize_lines:
        for path in paths:
            content = path.read_bytes()
            normalized = content.replace(b"\r\n", b"\n")
            if normalized != content:
                path.write_bytes(normalized)
    errors = []
    python_files = 0
    records = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        content = path.read_bytes()
        text = content.decode("utf-8-sig")
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=relative)
                python_files += 1
            except SyntaxError as exc:
                errors.append(f"{relative}: {exc}")
        if not relative.startswith("vendor/"):
            for label, pattern in PRIVATE_PATTERNS:
                if pattern.search(text):
                    errors.append(f"{relative}: {label}")
        if len(content) > 10 * 1024**2:
            errors.append(f"Unexpectedly large source file: {relative}")
        if relative != "docs/release_manifest.json":
            records.append({"path": relative, "bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest()})
    if errors:
        raise SystemExit("Release audit failed:\n" + "\n".join(errors))
    if args.write_manifest:
        (ROOT / "docs/release_manifest.json").write_text(
            json.dumps({"scope": "source-only; manifest excludes itself", "files": records}, indent=2) + "\n",
            encoding="utf-8")
        paths = list(source_files())
    if args.archive:
        destination = args.archive.resolve()
        if destination.is_relative_to(ROOT):
            raise ValueError("Place the ZIP outside the source tree")
        if destination.exists():
            raise FileExistsError("Refusing to overwrite an existing archive; use a new filename")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                relative = path.relative_to(ROOT).as_posix()
                info = zipfile.ZipInfo("hyperspectral-diffraction-radial-fingerprinting/" + relative,
                                       date_time=(2026, 9, 30, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
                data = path.read_bytes()
                if path.suffix == ".sh" and b"\r\n" in data:
                    raise ValueError("Normalize shell-script lines with --normalize-lines before archiving")
                archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(destination) as archive:
            assert archive.testzip() is None
            assert len(archive.infolist()) == len(paths)
        print(f"ZIP: {destination.name}; {destination.stat().st_size:,} bytes")
    print(f"PASS: {len(paths)} source files; {python_files} Python files parsed; no scanned private patterns")


if __name__ == "__main__":
    main()
