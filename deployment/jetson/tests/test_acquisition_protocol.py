#!/usr/bin/env python3
"""Hardware-free tests for the serial JPEG framing and scan geometry."""

from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve()
SCRIPT_CANDIDATES = [
    HERE.parents[1] / "acquisition" / "acquire_and_infer.py",
    HERE.parent / "jetson_acquire_and_infer.py",
]
SCRIPT = next((candidate for candidate in SCRIPT_CANDIDATES if candidate.exists()), None)
if SCRIPT is None:
    raise FileNotFoundError("acquire_and_infer.py was not found")
SPEC = importlib.util.spec_from_file_location("acquire_and_infer", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeSerial:
    def __init__(self, jpeg: bytes) -> None:
        self.jpeg = jpeg
        self.buffer = bytearray()
        self.writes: list[bytes] = []

    @property
    def in_waiting(self) -> int:
        return len(self.buffer)

    def reset_input_buffer(self) -> None:
        self.buffer.clear()

    def reset_output_buffer(self) -> None:
        pass

    def write(self, data: bytes) -> int:
        self.writes.append(data)
        if data == b"s":
            self.buffer.extend(b"noise\xaa\x55" + struct.pack("<L", len(self.jpeg)) + self.jpeg)
        return len(data)

    def flush(self) -> None:
        pass

    def read(self, size: int) -> bytes:
        chunk = bytes(self.buffer[:size])
        del self.buffer[:size]
        return chunk


class AcquisitionProtocolTests(unittest.TestCase):
    def test_scan_geometry_matches_existing_dataset(self) -> None:
        positions = MODULE.expected_positions(400, -900000, 4000)
        wavelengths = MODULE.expected_wavelengths(400, 400.0, 800.0)
        self.assertEqual(positions[0], -896000)
        self.assertEqual(positions[-1], 700000)
        self.assertEqual(len(set(positions)), 400)
        self.assertAlmostEqual(float(wavelengths[0]), 400.0)
        self.assertAlmostEqual(float(wavelengths[-1]), 800.0)

    def test_camera_framing_and_block_acknowledgement(self) -> None:
        image = np.full((12, 16, 3), 127, dtype=np.uint8)
        success, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(success)
        connection = FakeSerial(encoded.tobytes())
        camera = MODULE.SerialJpegCamera(
            connection,
            block_size=37,
            header_timeout=0.2,
            block_timeout=0.2,
            maximum_jpeg_bytes=100_000,
        )
        payload, decoded, attempts = camera.capture(maximum_attempts=1)
        self.assertEqual(payload, encoded.tobytes())
        self.assertEqual(decoded.shape[:2], image.shape[:2])
        self.assertEqual(attempts, 1)
        expected_acks = (len(payload) + 36) // 37
        self.assertEqual(connection.writes[0], b"s")
        self.assertEqual(connection.writes.count(b"k"), expected_acks)

    def test_complete_sequence_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            image = np.full((9, 13), 80, dtype=np.uint8)
            for position in MODULE.expected_positions(4, -4, 1):
                cv2.imwrite(str(folder / f"pos_{position}.jpg"), image)
            self.assertEqual(MODULE.validate_sequence(folder, 4, -4, 1), (13, 9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
