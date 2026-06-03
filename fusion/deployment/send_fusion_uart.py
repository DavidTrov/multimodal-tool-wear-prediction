"""
send_fusion_uart.py — Stream sensor CSV (per-channel) + tool flank image to
FRDM-MCXN947 and read back the fusion-model wear prediction.

The MCU protocol (per-channel):
  1. MCU: "READY"
  2. For each channel 0..4:
     a. MCU: "SEND_CH{n}"
     b. Host: one float per line (the channel column), then "END"
     c. MCU: computes CWT
  3. MCU: TFLite init, prints quantisation params, "SEND_IMAGE"
  4. Host: sends 150,528 raw INT8 bytes (3×224×224 NCHW)
  5. MCU: "Predicted tool wear: XX.X um"

Usage:
    python experiments/phase6_deployment/send_fusion_uart.py \\
        --port /dev/tty.usbmodemXXXX \\
        --csv  data/raw/Set4/sensordata/<file>.csv \\
        --image data/raw/Set4/<flank_image>.jpg \\
        --set-id 4

Optional flags:
    --baud 921600        Higher baud rate (default 115200)
    --no-mask            Skip cutting-mask extraction
    --img-scale FLOAT    Override image quantisation scale   (default: 0.020787)
    --img-zp    INT      Override image quantisation zero_point (default: 0)
"""

import argparse
import ast
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import serial
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.aircut_mask import cutting_mask, extract_cutting_signal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSOR_COLS    = ["acc", "acoustic", "fx", "fy", "fz"]
ALL_COLS       = ["acc", "acoustic", "fx", "fy", "fz", "time"]
BAUD           = 115200
TIMEOUT_READY  = 30    # seconds to wait for MCU "READY"
TIMEOUT_CWT    = 60    # seconds to wait for each "SEND_CH{n}" after previous channel
TIMEOUT_IMAGE  = 120   # seconds to wait for "SEND_IMAGE" after last CWT
TIMEOUT_PRED   = 120   # seconds to wait for prediction after image

# Image quantisation params from the TFLite model
# (fusion_int8_qat_nxp_io.tflite, input index 0 — see io_quant.json):
#   scale = 0.020787402987480164,  zero_point = 0
DEFAULT_IMG_SCALE = 0.020787402987480164
DEFAULT_IMG_ZP    = 0

# ImageNet normalisation statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Sets CSV path (relative to repo root)
SETS_CSV = ROOT / "data" / "raw" / "sets.csv"

# --reset support: reboot the target over SWD (via the debug probe) so the MCU
# re-emits its one-shot "READY" while the host already has the port open. The
# physical RESET button can drop the MCU-Link VCOM; an SWD reflash-and-reset
# does not. flash "load" resets the target on completion.
DEFAULT_LINKSERVER = "/Applications/LinkServer_25.6.131/LinkServer"
DEFAULT_DEVICE     = "MCXN947:FRDM-MCXN947"
DEFAULT_AXF        = (
    "/Users/david/Projects/University/Maastricht University/"
    "frdmmcxn947_tflm_cifar10_cm33_core0/Debug/"
    "frdmmcxn947_tflm_cifar10_cm33_core0.axf"
)


def reset_target(linkserver: str, device: str, axf: str) -> subprocess.Popen:
    """Reboot the target via the debug probe, concurrently with READY draining.

    Runs LinkServer flash-load in the background: it programs the image (no-op
    if unchanged) and issues a system reset on completion, which reboots the
    firmware so it re-prints "READY". Returns the Popen so the caller can drain
    the serial port while this runs."""
    cmd = [linkserver, "flash", device, "load", axf]
    print("Resetting target over SWD (reboot to re-emit READY)...", flush=True)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

# ---------------------------------------------------------------------------
# Helper: load crop coordinates from sets.csv
# ---------------------------------------------------------------------------

def load_crop_coords(set_id: int):
    df = pd.read_csv(SETS_CSV, index_col=0)
    row_name = f"Set {set_id}"
    if row_name not in df.index:
        raise KeyError(f"Set {set_id} not found in {SETS_CSV}")
    crop_str = df.loc[row_name, "crop"]
    coords   = ast.literal_eval(f"({crop_str})")
    return tuple(int(c) for c in coords)


# ---------------------------------------------------------------------------
# Image preprocessing + INT8 quantisation
# ---------------------------------------------------------------------------

def preprocess_image_int8(
    img_path: Path,
    crop_coords: tuple,
    img_scale: float = DEFAULT_IMG_SCALE,
    img_zp:    int   = DEFAULT_IMG_ZP,
) -> bytes:
    """
    Crop → resize 224×224 → ImageNet-normalise → quantise to INT8 → raw bytes.
    Returns 150,528 bytes in NCHW layout (channels first: 3 × 224 × 224).
    Model input is [1, 3, 224, 224] INT8 NCHW.
    """
    img = Image.open(img_path).convert("RGB")
    img = img.crop(crop_coords)
    img = img.resize((224, 224), Image.BILINEAR)

    x = np.array(img, dtype=np.float32) / 255.0         # (224, 224, 3) HWC

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    x    = (x - mean) / std                              # (224, 224, 3) HWC

    x = x.transpose(2, 0, 1)                             # (3, 224, 224) CHW → NCHW

    q = np.round(x / img_scale).astype(np.int32) + img_zp
    q = np.clip(q, -128, 127).astype(np.int8)

    return q.tobytes()   # 150,528 bytes in NCHW order


# ---------------------------------------------------------------------------
# Sensor CSV loading
# ---------------------------------------------------------------------------

def load_cutting_segment(csv_path: Path) -> pd.DataFrame:
    """Load CSV, apply cutting mask, return 5-column cutting-segment DataFrame."""
    raw = pd.read_csv(csv_path, header=None)
    if raw.shape[1] < 5:
        raise ValueError(f"Expected ≥5 columns, got {raw.shape[1]}")
    raw.columns = ALL_COLS[: raw.shape[1]]

    fx   = raw["fx"].to_numpy(dtype=np.float64)
    fy   = raw["fy"].to_numpy(dtype=np.float64)
    fz   = raw["fz"].to_numpy(dtype=np.float64)
    mask = cutting_mask(fx, fy, fz)

    channels = {col: extract_cutting_signal(raw[col].to_numpy(dtype=np.float64), mask)
                for col in SENSOR_COLS}
    n = len(channels["acc"])
    print(f"Cutting segment: {n} samples ({n / 1625:.1f} s at 1625 Hz)")
    return pd.DataFrame(channels)


# ---------------------------------------------------------------------------
# UART helpers
# ---------------------------------------------------------------------------

def drain_until(ser, expected: str, timeout: float) -> str:
    """
    Read lines from MCU until `expected` appears in a line, or timeout.
    Prints all received lines. Returns the matched line.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if line:
            print(f"  MCU: {line}")
        if expected in line:
            return line
    raise TimeoutError(f"Timed out ({timeout:.0f}s) waiting for '{expected}'")


def send_channel(ser, col_data, ch_idx: int, col_name: str, timeout_next):
    """
    Send one channel column as lines of floats + 'END'.
    Then drain MCU output until the next expected prompt.
    Returns when MCU is ready for the next step.
    """
    n = len(col_data)
    print(f"  Sending {col_name} ({n} samples)...")
    chunk_lines = 500
    for i, val in enumerate(col_data):
        ser.write(f"{val:.8f}\n".encode("ascii"))
        if (i + 1) % chunk_lines == 0:
            # Brief yield to avoid USB CDC buffer saturation at high baud rates
            time.sleep(0.005)
    ser.write(b"END\n")
    ser.flush()
    print(f"  {n} values + END sent.")
    # The caller drains until the next SEND_CH or SEND_IMAGE prompt


# ---------------------------------------------------------------------------
# Main UART round-trip
# ---------------------------------------------------------------------------

def send_and_receive(
    port:      str,
    baud:      int,
    df:        pd.DataFrame,
    img_bytes: bytes,
    reset:     bool = False,
    linkserver: str = DEFAULT_LINKSERVER,
    device:    str = DEFAULT_DEVICE,
    axf:       str = DEFAULT_AXF,
) -> str:
    """
    Full per-channel protocol:
      1. Wait for "READY"
      2. For each channel: wait for "SEND_CH{n}", send column, drain
      3. Wait for "SEND_IMAGE"
      4. Send image bytes
      5. Wait for prediction line, return it
    """
    img_size = len(img_bytes)
    print(f"Opening {port} at {baud} baud...")
    with serial.Serial(port, baud, timeout=5.0) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # ── 1. Wait for READY ────────────────────────────────────────────
        # Fire the reset AFTER the port is open and draining starts, so the
        # one-shot "READY" lands while we're listening (no boot-time race).
        if reset:
            reset_target(linkserver, device, axf)
        print("Waiting for MCU READY...", flush=True)
        drain_until(ser, "READY", TIMEOUT_READY)

        # ── 2. Per-channel data send ──────────────────────────────────────
        for ch_idx, col_name in enumerate(SENSOR_COLS):
            # Wait for MCU to request this channel
            print(f"Waiting for SEND_CH{ch_idx}...", flush=True)
            drain_until(ser, f"SEND_CH{ch_idx}", TIMEOUT_CWT)

            # Send channel data
            send_channel(ser, df[col_name], ch_idx, col_name, TIMEOUT_CWT)

        # ── 3. Wait for SEND_IMAGE ────────────────────────────────────────
        print(f"Waiting for TFLite init + SEND_IMAGE (up to {TIMEOUT_IMAGE}s)...",
              flush=True)
        drain_until(ser, "SEND_IMAGE", TIMEOUT_IMAGE)

        # ── 4. Send image bytes ───────────────────────────────────────────
        print(f"Sending image ({img_size} bytes = {img_size/1024:.1f} KB)...")
        t0         = time.time()
        chunk_size = 512
        sent       = 0
        while sent < img_size:
            end   = min(sent + chunk_size, img_size)
            ser.write(img_bytes[sent:end])
            sent  = end
            if sent % 4096 == 0:
                time.sleep(0.001)   # yield every 4 KB

        ser.flush()
        elapsed = time.time() - t0
        print(f"  Image sent in {elapsed:.1f}s  "
              f"({img_size / elapsed / 1024:.0f} KB/s)")

        # ── 5. Wait for prediction ────────────────────────────────────────
        print(f"Waiting for prediction (up to {TIMEOUT_PRED}s)...", flush=True)
        result = drain_until(ser, "Predicted tool wear", TIMEOUT_PRED)
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Send per-channel sensor CSV + image to FRDM-MCXN947, read wear prediction"
    )
    parser.add_argument("--port",      required=True,
                        help="Serial port, e.g. /dev/tty.usbmodemXXXX")
    parser.add_argument("--csv",       required=True,
                        help="Path to raw sensor CSV (5 channels)")
    parser.add_argument("--image",     required=True,
                        help="Path to tool flank image (JPG/PNG)")
    parser.add_argument("--set-id",    type=int, default=None,
                        help="Set ID (1-13) for crop lookup in sets.csv")
    parser.add_argument("--baud",      type=int, default=BAUD,
                        help=f"UART baud rate (default {BAUD}; try 921600 for faster transfer)")
    parser.add_argument("--no-mask",   action="store_true",
                        help="Skip cutting-mask extraction (send full sensor CSV)")
    parser.add_argument("--img-scale", type=float, default=DEFAULT_IMG_SCALE,
                        help=f"Image INT8 quantisation scale (default {DEFAULT_IMG_SCALE})")
    parser.add_argument("--img-zp",    type=int,   default=DEFAULT_IMG_ZP,
                        help=f"Image INT8 quantisation zero_point (default {DEFAULT_IMG_ZP})")
    parser.add_argument("--reset",     action="store_true",
                        help="Reboot the target over the debug probe after opening "
                             "the port, so the MCU re-emits READY (avoids the boot race)")
    parser.add_argument("--linkserver", default=DEFAULT_LINKSERVER,
                        help=f"LinkServer binary path (default {DEFAULT_LINKSERVER})")
    parser.add_argument("--device",    default=DEFAULT_DEVICE,
                        help=f"LinkServer device:board (default {DEFAULT_DEVICE})")
    parser.add_argument("--axf",       default=DEFAULT_AXF,
                        help="Path to the .axf to reflash/reset with --reset")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    img_path = Path(args.image)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    if not img_path.exists():
        sys.exit(f"Image not found: {img_path}")

    # ── Sensor CSV ──────────────────────────────────────────────────────
    if args.no_mask:
        raw = pd.read_csv(csv_path, header=None)
        raw.columns = ALL_COLS[: raw.shape[1]]
        df = raw[SENSOR_COLS]
        print(f"Full CSV: {len(df)} rows (no cutting mask)")
    else:
        df = load_cutting_segment(csv_path)

    print(f"Channels: {list(df.columns)}, samples per channel: {len(df)}")
    if len(df) < 64:
        sys.exit(f"ERROR: too few samples ({len(df)} < 64)")

    # ── Image ────────────────────────────────────────────────────────────
    if args.set_id is not None:
        crop_coords = load_crop_coords(args.set_id)
        print(f"Set {args.set_id} crop: {crop_coords}")
    else:
        img_tmp = Image.open(img_path)
        crop_coords = (0, 0, img_tmp.width, img_tmp.height)
        print("Warning: no --set-id given; using full image without crop.")

    img_bytes = preprocess_image_int8(
        img_path, crop_coords, args.img_scale, args.img_zp)
    print(f"Image preprocessed: {len(img_bytes)} bytes  "
          f"(scale={args.img_scale}, zp={args.img_zp})")

    # ── UART round-trip ──────────────────────────────────────────────────
    result = send_and_receive(args.port, args.baud, df, img_bytes,
                              reset=args.reset, linkserver=args.linkserver,
                              device=args.device, axf=args.axf)
    print(f"\n=== Result ===\n{result}")


if __name__ == "__main__":
    main()
