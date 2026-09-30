#!/usr/bin/env python3
"""Acquire one 400-band scan from the serial camera/stage and run inference on Firefly.

The acquisition protocol is preserved from the original Windows controller:
the camera receives ``s``, returns ``AA 55`` + a little-endian uint32 JPEG size,
and waits for ``k`` after every data block. The stage is homed once and then
moved by 4000 controller units before every captured band.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # Existing-sequence validation and protocol tests do not need pyserial.
    serial = None
    list_ports = None


ROOT = Path(__file__).resolve().parent
if ROOT.name == "acquisition":
    ROOT = ROOT.parent
CMD_INIT_POSITION = bytes.fromhex("a5 53 01 82 ff f2 44 60 01 19")
CMD_STEP_4000 = bytes.fromhex("a5 53 01 83 00 00 0f a0 01 34")
POSITION_PATTERN = re.compile(r"^pos_(-?\d+)\.jpg$", re.IGNORECASE)


@dataclass
class FrameRecord:
    index: int
    position: int
    wavelength_nm: float
    filename: str
    bytes: int
    attempts: int
    capture_seconds: float


def utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def scan_identifier() -> str:
    return datetime.now().strftime("scan_%Y%m%d_%H%M%S")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def expected_positions(total_images: int, initial_position: int, step_position: int) -> list[int]:
    return [initial_position + step_position * (index + 1) for index in range(total_images)]


def expected_wavelengths(total_images: int, minimum_nm: float, maximum_nm: float) -> np.ndarray:
    if total_images < 2:
        return np.array([minimum_nm], dtype=np.float64)
    return np.linspace(minimum_nm, maximum_nm, total_images, dtype=np.float64)


def clean_serial(connection: serial.Serial) -> None:
    connection.reset_input_buffer()
    connection.reset_output_buffer()


def read_exactly(connection: serial.Serial, size: int, timeout_seconds: float) -> bytes:
    data = bytearray()
    deadline = time.monotonic() + timeout_seconds
    while len(data) < size:
        if time.monotonic() >= deadline:
            break
        waiting = connection.in_waiting
        if waiting:
            data.extend(connection.read(min(size - len(data), waiting)))
        else:
            time.sleep(0.001)
    return bytes(data)


class SerialJpegCamera:
    def __init__(
        self,
        connection: serial.Serial,
        block_size: int,
        header_timeout: float,
        block_timeout: float,
        maximum_jpeg_bytes: int,
    ) -> None:
        self.connection = connection
        self.block_size = block_size
        self.header_timeout = header_timeout
        self.block_timeout = block_timeout
        self.maximum_jpeg_bytes = maximum_jpeg_bytes

    def _receive_once(self) -> tuple[bytes, np.ndarray]:
        clean_serial(self.connection)
        self.connection.write(b"s")
        self.connection.flush()

        previous = b""
        deadline = time.monotonic() + self.header_timeout
        while time.monotonic() < deadline:
            if self.connection.in_waiting:
                current = self.connection.read(1)
                if previous == b"\xaa" and current == b"\x55":
                    break
                previous = current
            else:
                time.sleep(0.001)
        else:
            raise TimeoutError("camera frame header AA55 was not received")

        size_data = read_exactly(self.connection, 4, self.block_timeout)
        if len(size_data) != 4:
            raise TimeoutError("camera JPEG size field was incomplete")
        image_size = struct.unpack("<L", size_data)[0]
        if not 0 < image_size <= self.maximum_jpeg_bytes:
            raise ValueError(f"invalid camera JPEG size: {image_size}")

        image_data = bytearray()
        while len(image_data) < image_size:
            chunk_size = min(self.block_size, image_size - len(image_data))
            chunk = read_exactly(self.connection, chunk_size, self.block_timeout)
            if len(chunk) != chunk_size:
                raise TimeoutError(
                    f"camera JPEG stopped at {len(image_data) + len(chunk)}/{image_size} bytes"
                )
            image_data.extend(chunk)
            self.connection.write(b"k")
            self.connection.flush()

        encoded = bytes(image_data)
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("camera payload is not a decodable JPEG")
        return encoded, decoded

    def capture(self, maximum_attempts: int) -> tuple[bytes, np.ndarray, int]:
        last_error: Exception | None = None
        for attempt in range(1, maximum_attempts + 1):
            try:
                encoded, decoded = self._receive_once()
                return encoded, decoded, attempt
            except Exception as error:
                last_error = error
                print(f"[camera] attempt {attempt}/{maximum_attempts} failed: {error}", flush=True)
                time.sleep(0.15)
        raise RuntimeError(f"camera capture failed after {maximum_attempts} attempts: {last_error}")


class TranslationStage:
    def __init__(self, connection: serial.Serial) -> None:
        self.connection = connection

    def send(self, command: bytes) -> None:
        clean_serial(self.connection)
        self.connection.write(command)
        self.connection.flush()

    def home(self, wait_seconds: float) -> None:
        print(f"[stage] homing; fixed wait {wait_seconds:.1f} s", flush=True)
        self.send(CMD_INIT_POSITION)
        time.sleep(wait_seconds)

    def step(self, command_delay: float, settle_seconds: float) -> None:
        self.send(CMD_STEP_4000)
        time.sleep(command_delay)
        time.sleep(settle_seconds)


def available_serial_ports() -> list[dict[str, object]]:
    if list_ports is None:
        raise RuntimeError("pyserial is required: python3 -m pip install pyserial")
    rows: list[dict[str, object]] = []
    for port in list_ports.comports():
        rows.append(
            {
                "device": port.device,
                "description": port.description,
                "manufacturer": port.manufacturer,
                "serial_number": port.serial_number,
                "vid": port.vid,
                "pid": port.pid,
            }
        )
    return rows


def print_serial_ports() -> None:
    rows = available_serial_ports()
    if not rows:
        print("No serial devices detected. Connect the camera controller and stage, then retry.")
        return
    for row in rows:
        vid_pid = ""
        if row["vid"] is not None and row["pid"] is not None:
            vid_pid = f" VID:PID={row['vid']:04x}:{row['pid']:04x}"
        serial_number = f" serial={row['serial_number']}" if row["serial_number"] else ""
        print(f"{row['device']}: {row['description']}{vid_pid}{serial_number}")


def require_display(option_name: str) -> None:
    has_linux_display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    if sys.platform.startswith("linux") and not has_linux_display:
        raise RuntimeError(
            f"{option_name} requires a local desktop display. Run from the RK3588 screen or omit it over SSH."
        )


def configure_display_window(title: str) -> None:
    flags = cv2.WINDOW_NORMAL | getattr(cv2, "WINDOW_GUI_NORMAL", 0)
    cv2.namedWindow(title, flags)
    fullscreen = os.environ.get("HSI_FULLSCREEN", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }
    if fullscreen:
        cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(title, 1280, 900)
    cv2.waitKey(1)


def preview_camera(camera: SerialJpegCamera, maximum_attempts: int) -> None:
    require_display("--preview")
    window = "HSI camera preview - Q/ESC to continue"
    configure_display_window(window)
    frame_number = 0
    print("[preview] focus the optical image, then press Q or ESC", flush=True)
    try:
        while True:
            _, frame, _ = camera.capture(maximum_attempts)
            frame_number += 1
            shown = frame.copy()
            cv2.putText(
                shown,
                f"Preview {frame_number} | Q/ESC: continue",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)


def preview_camera_only(arguments: argparse.Namespace) -> None:
    if serial is None:
        raise RuntimeError("pyserial is required: python3 -m pip install pyserial")
    if not arguments.camera_port:
        raise ValueError("--camera-port is required with --preview-only")
    print(f"[hardware] opening camera {arguments.camera_port} @ {arguments.camera_baud}")
    connection = serial.Serial(
        arguments.camera_port,
        arguments.camera_baud,
        timeout=0.1,
        write_timeout=2.0,
    )
    try:
        camera = SerialJpegCamera(
            connection,
            block_size=arguments.block_size,
            header_timeout=arguments.header_timeout,
            block_timeout=arguments.block_timeout,
            maximum_jpeg_bytes=arguments.maximum_jpeg_bytes,
        )
        preview_camera(camera, arguments.frame_retries)
    finally:
        connection.close()
        cv2.destroyAllWindows()
    print("[PASS] camera preview protocol test complete; the stage was not opened or moved")


def show_scan_progress(frame: np.ndarray, index: int, total: int, wavelength: float, position: int) -> None:
    shown = frame.copy()
    label = f"Band {index}/{total} | {wavelength:.1f} nm | pos_{position}"
    cv2.rectangle(shown, (0, 0), (shown.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(
        shown,
        label,
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("HSI wavelength scan", shown)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        raise KeyboardInterrupt("scan cancelled from display")


def write_progress_manifest(
    path: Path,
    scan_id: str,
    status: str,
    arguments: argparse.Namespace,
    records: list[FrameRecord],
    started_at: str,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "scan_id": scan_id,
        "status": status,
        "started_at": started_at,
        "updated_at": utc_timestamp(),
        "protocol": {
            "camera_request_hex": "73",
            "camera_header_hex": "aa55",
            "camera_ack_hex": "6b",
            "stage_home_hex": CMD_INIT_POSITION.hex(" "),
            "stage_step_hex": CMD_STEP_4000.hex(" "),
        },
        "configuration": {
            "camera_port": arguments.camera_port,
            "camera_baud": arguments.camera_baud,
            "stage_port": arguments.stage_port,
            "stage_baud": arguments.stage_baud,
            "total_images": arguments.total_images,
            "initial_position": arguments.initial_position,
            "step_position": arguments.step_position,
            "wavelength_min_nm": arguments.wavelength_min,
            "wavelength_max_nm": arguments.wavelength_max,
            "settle_seconds": arguments.settle_seconds,
        },
        "completed_frames": len(records),
        "frames": [asdict(record) for record in records],
    }
    if error:
        payload["error"] = error
    atomic_json(path, payload)


def validate_sequence(
    folder: Path,
    total_images: int,
    initial_position: int,
    step_position: int,
) -> tuple[int, int]:
    expected = expected_positions(total_images, initial_position, step_position)
    observed: dict[int, Path] = {}
    for path in folder.glob("pos_*.jpg"):
        match = POSITION_PATTERN.match(path.name)
        if match:
            observed[int(match.group(1))] = path
    missing = [position for position in expected if position not in observed]
    extras = sorted(set(observed) - set(expected))
    if missing or extras:
        raise ValueError(
            f"sequence position mismatch: missing={missing[:8]} ({len(missing)} total), "
            f"extra={extras[:8]} ({len(extras)} total)"
        )

    shape: tuple[int, int] | None = None
    for position in expected:
        image = cv2.imread(str(observed[position]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"unreadable JPEG: {observed[position]}")
        current_shape = (int(image.shape[1]), int(image.shape[0]))
        if shape is None:
            shape = current_shape
        elif current_shape != shape:
            raise ValueError(
                f"image shape mismatch at {observed[position].name}: {current_shape} != {shape}"
            )
    assert shape is not None
    return shape


def acquire_sequence(arguments: argparse.Namespace, root: Path) -> tuple[Path, dict[str, Any]]:
    if serial is None:
        raise RuntimeError("pyserial is required: python3 -m pip install pyserial")
    if not arguments.camera_port or not arguments.stage_port:
        raise ValueError("--camera-port and --stage-port are required for hardware acquisition")
    if arguments.camera_port == arguments.stage_port:
        raise ValueError("camera and stage ports must be different")
    if arguments.show_scan_window:
        require_display("--show-scan-window")

    acquisition_root = (root / arguments.acquisition_dir).resolve()
    acquisition_root.mkdir(parents=True, exist_ok=True)
    scan_id = arguments.scan_name or scan_identifier()
    incomplete = acquisition_root / f"{scan_id}.partial"
    complete = acquisition_root / scan_id
    if incomplete.exists() or complete.exists():
        raise FileExistsError(f"scan destination already exists for {scan_id}")
    incomplete.mkdir(parents=True)

    started_at = utc_timestamp()
    started_clock = time.monotonic()
    records: list[FrameRecord] = []
    manifest_path = incomplete / "acquisition_manifest.json"
    write_progress_manifest(manifest_path, scan_id, "initializing", arguments, records, started_at)
    positions = expected_positions(
        arguments.total_images, arguments.initial_position, arguments.step_position
    )
    wavelengths = expected_wavelengths(
        arguments.total_images, arguments.wavelength_min, arguments.wavelength_max
    )

    camera_serial: serial.Serial | None = None
    stage_serial: serial.Serial | None = None
    try:
        print(f"[hardware] opening camera {arguments.camera_port} @ {arguments.camera_baud}")
        camera_serial = serial.Serial(
            arguments.camera_port,
            arguments.camera_baud,
            timeout=0.1,
            write_timeout=2.0,
        )
        print(f"[hardware] opening stage {arguments.stage_port} @ {arguments.stage_baud}")
        stage_serial = serial.Serial(
            arguments.stage_port,
            arguments.stage_baud,
            timeout=0.1,
            write_timeout=2.0,
        )
        camera = SerialJpegCamera(
            camera_serial,
            block_size=arguments.block_size,
            header_timeout=arguments.header_timeout,
            block_timeout=arguments.block_timeout,
            maximum_jpeg_bytes=arguments.maximum_jpeg_bytes,
        )
        stage = TranslationStage(stage_serial)

        if arguments.preview:
            preview_camera(camera, arguments.frame_retries)
        if not arguments.assume_yes:
            answer = input("Start the 400-band scan now? [y/N]: ").strip().lower()
            if answer not in {"y", "yes"}:
                raise KeyboardInterrupt("scan cancelled before stage movement")

        stage.home(arguments.home_wait_seconds)
        write_progress_manifest(manifest_path, scan_id, "scanning", arguments, records, started_at)
        if arguments.show_scan_window:
            configure_display_window("HSI wavelength scan")

        for offset, (position, wavelength) in enumerate(zip(positions, wavelengths), start=1):
            stage.step(arguments.command_delay, arguments.settle_seconds)
            capture_started = time.monotonic()
            encoded, frame, attempts = camera.capture(arguments.frame_retries)
            capture_seconds = time.monotonic() - capture_started
            filename = f"pos_{position}.jpg"
            temporary_file = incomplete / f"{filename}.part"
            final_file = incomplete / filename
            temporary_file.write_bytes(encoded)
            os.replace(temporary_file, final_file)
            records.append(
                FrameRecord(
                    index=offset,
                    position=position,
                    wavelength_nm=float(wavelength),
                    filename=filename,
                    bytes=len(encoded),
                    attempts=attempts,
                    capture_seconds=capture_seconds,
                )
            )
            print(
                f"[scan] {offset:03d}/{arguments.total_images} "
                f"{wavelength:7.2f} nm pos_{position} "
                f"{len(encoded) / 1024:7.1f} KiB ({capture_seconds:.2f} s)",
                flush=True,
            )
            if offset == 1 or offset % arguments.manifest_interval == 0:
                write_progress_manifest(
                    manifest_path, scan_id, "scanning", arguments, records, started_at
                )
            if arguments.show_scan_window:
                show_scan_progress(
                    frame, offset, arguments.total_images, float(wavelength), position
                )

        width, height = validate_sequence(
            incomplete,
            arguments.total_images,
            arguments.initial_position,
            arguments.step_position,
        )
        completed_seconds = time.monotonic() - started_clock
        write_progress_manifest(manifest_path, scan_id, "complete", arguments, records, started_at)
        os.replace(incomplete, complete)
        summary = {
            "scan_id": scan_id,
            "input_dir": str(complete),
            "frames": len(records),
            "image_width": width,
            "image_height": height,
            "acquisition_seconds": completed_seconds,
            "first_position": positions[0],
            "last_position": positions[-1],
            "wavelength_min_nm": arguments.wavelength_min,
            "wavelength_max_nm": arguments.wavelength_max,
        }
        print(f"[PASS] acquisition complete: {complete}")
        return complete, summary
    except BaseException as error:
        write_progress_manifest(
            manifest_path,
            scan_id,
            "failed",
            arguments,
            records,
            started_at,
            error=str(error),
        )
        print(f"[FAIL] incomplete acquisition retained at {incomplete}", file=sys.stderr)
        raise
    finally:
        if camera_serial is not None:
            camera_serial.close()
        if stage_serial is not None:
            stage_serial.close()
        if arguments.show_scan_window or arguments.preview:
            cv2.destroyAllWindows()


def run_inference(
    root: Path,
    input_dir: Path,
    output_dir: Path,
    arguments: argparse.Namespace,
) -> float:
    launcher = root / "run_rk3588_hsi_once.sh"
    if not launcher.exists():
        raise FileNotFoundError(launcher)
    command = ["bash", str(launcher), str(input_dir), str(output_dir)]
    if arguments.no_save_cubes:
        command.append("--no-save-cubes")
    if arguments.minimal_output:
        command.append("--minimal-output")
    if arguments.show_inference_windows:
        require_display("--show-inference-windows")
        command.extend(["--show-windows", "--popup-delay-ms", str(arguments.popup_delay_ms)])
    if arguments.pipeline_args:
        command.extend(arguments.pipeline_args)
    print("[inference] " + " ".join(command), flush=True)
    started = time.monotonic()
    subprocess.run(command, cwd=root, check=True)
    elapsed = time.monotonic() - started
    required_outputs = [
        output_dir / "final_result.png",
        output_dir / "pipeline_summary.json",
        output_dir / "detections_and_classification.csv",
    ]
    missing = [str(path) for path in required_outputs if not path.exists()]
    if missing:
        raise RuntimeError(f"inference returned without required outputs: {missing}")
    return elapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serial camera + translation-stage acquisition followed by RKNN/CPU pose detection, "
            "ROI classification, counting and concentration estimation."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--camera-port")
    parser.add_argument("--stage-port")
    parser.add_argument("--camera-baud", type=int, default=115200)
    parser.add_argument("--stage-baud", type=int, default=115200)
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--existing-sequence", type=Path)
    parser.add_argument("--acquisition-dir", type=Path, default=Path("acquisitions"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--scan-name")
    parser.add_argument("--total-images", type=int, default=400)
    parser.add_argument("--initial-position", type=int, default=-900000)
    parser.add_argument("--step-position", type=int, default=4000)
    parser.add_argument("--wavelength-min", type=float, default=400.0)
    parser.add_argument("--wavelength-max", type=float, default=800.0)
    parser.add_argument("--home-wait-seconds", type=float, default=6.0)
    parser.add_argument("--command-delay", type=float, default=0.05)
    parser.add_argument("--settle-seconds", type=float, default=0.5)
    parser.add_argument("--frame-retries", type=int, default=3)
    parser.add_argument("--header-timeout", type=float, default=5.0)
    parser.add_argument("--block-timeout", type=float, default=2.0)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--maximum-jpeg-bytes", type=int, default=8_000_000)
    parser.add_argument("--manifest-interval", type=int, default=10)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Test live camera framing and decoding without opening or moving the stage.",
    )
    parser.add_argument("--show-scan-window", action="store_true")
    parser.add_argument("--show-inference-windows", action="store_true")
    parser.add_argument("--popup-delay-ms", type=int, default=1500)
    parser.add_argument("--assume-yes", action="store_true")
    parser.add_argument("--no-inference", action="store_true")
    parser.add_argument("--no-save-cubes", action="store_true")
    parser.add_argument(
        "--minimal-output",
        action="store_true",
        help="Keep final and tabular outputs while skipping redundant preview files.",
    )
    parser.add_argument(
        "--pipeline-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed verbatim to rk3588_hsi_end_to_end.py.",
    )
    return parser


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.total_images <= 0:
        raise ValueError("--total-images must be positive")
    if arguments.frame_retries <= 0:
        raise ValueError("--frame-retries must be positive")
    if arguments.block_size <= 0 or arguments.maximum_jpeg_bytes <= 0:
        raise ValueError("camera byte limits must be positive")
    if arguments.manifest_interval <= 0:
        raise ValueError("--manifest-interval must be positive")


def main() -> None:
    arguments = build_parser().parse_args()
    validate_arguments(arguments)
    root = arguments.root.resolve()
    if arguments.list_ports:
        print_serial_ports()
        return
    if arguments.preview_only:
        preview_camera_only(arguments)
        return

    session_started = utc_timestamp()
    if arguments.existing_sequence:
        input_dir = arguments.existing_sequence.resolve()
        width, height = validate_sequence(
            input_dir,
            arguments.total_images,
            arguments.initial_position,
            arguments.step_position,
        )
        acquisition_summary: dict[str, Any] = {
            "mode": "existing_sequence",
            "input_dir": str(input_dir),
            "frames": arguments.total_images,
            "image_width": width,
            "image_height": height,
        }
        run_id = arguments.scan_name or f"{input_dir.name}_{datetime.now():%Y%m%d_%H%M%S}"
        print(f"[PASS] existing sequence validated: {input_dir}")
    else:
        input_dir, acquisition_summary = acquire_sequence(arguments, root)
        run_id = acquisition_summary["scan_id"]

    output_dir = (root / arguments.results_dir / run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    inference_seconds: float | None = None
    status = "acquisition_only" if arguments.no_inference else "complete"
    try:
        if not arguments.no_inference:
            inference_seconds = run_inference(root, input_dir, output_dir, arguments)
    except BaseException:
        status = "inference_failed"
        raise
    finally:
        summary = {
            "schema_version": 1,
            "status": status,
            "session_started_at": session_started,
            "session_finished_at": utc_timestamp(),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "acquisition": acquisition_summary,
            "inference_seconds": inference_seconds,
            "final_result": str(output_dir / "final_result.png"),
            "pipeline_summary": str(output_dir / "pipeline_summary.json"),
            "detections_csv": str(output_dir / "detections_and_classification.csv"),
        }
        atomic_json(output_dir / "acquire_and_infer_summary.json", summary)

    print(f"[PASS] end-to-end output: {output_dir}")
    if not arguments.no_inference:
        print(f"[PASS] final visualization: {output_dir / 'final_result.png'}")


if __name__ == "__main__":
    main()
