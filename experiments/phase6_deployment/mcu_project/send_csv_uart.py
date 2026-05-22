"""
send_csv_uart.py — Send a cutting-segment CSV to the FRDM-MCXN947 over UART
and read back the predicted tool wear.

Usage:
    python experiments/phase6_deployment/mcu_project/send_csv_uart.py \
        --port /dev/tty.usbmodemXXXX \
        --csv  data/raw/Set4/sensordata/<file>.csv \
        --baud 115200

The script:
  1. Waits for the MCU to print "READY"
  2. Applies the cutting mask (same as cwt_mcu_process_channel uses) and
     sends only the cutting-segment rows
  3. Sends "END" to signal completion
  4. Reads and prints the MCU's prediction

The MCU expects columns in order: acc,acoustic,fx,fy,fz (no header).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import serial

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.aircut_mask import cutting_mask, extract_cutting_signal

SENSOR_COLS = ["acc", "acoustic", "fx", "fy", "fz"]
ALL_COLS    = ["acc", "acoustic", "fx", "fy", "fz", "time"]
BAUD        = 115200
TIMEOUT_S   = 120   # seconds to wait for MCU response after sending END


def load_cutting_segment(csv_path: Path) -> pd.DataFrame:
    """Load CSV, apply cutting mask, return DataFrame with 5 columns."""
    raw = pd.read_csv(csv_path, header=None)
    if raw.shape[1] < 5:
        raise ValueError(f"Expected ≥5 columns, got {raw.shape[1]}")
    raw.columns = ALL_COLS[:raw.shape[1]]

    fx = raw["fx"].to_numpy(dtype=np.float64)
    fy = raw["fy"].to_numpy(dtype=np.float64)
    fz = raw["fz"].to_numpy(dtype=np.float64)
    mask = cutting_mask(fx, fy, fz)

    channels = {}
    for col in SENSOR_COLS:
        x = raw[col].to_numpy(dtype=np.float64)
        channels[col] = extract_cutting_signal(x, mask)

    n = len(channels["acc"])
    df = pd.DataFrame(channels)
    print(f"Cutting segment: {n} samples  "
          f"({n / 1625:.1f} s at 1625 Hz)")
    return df


def send_and_receive(port: str, baud: int, df: pd.DataFrame) -> str:
    """Open UART, wait for READY, stream CSV rows, return MCU response."""
    print(f"Opening {port} at {baud} baud...")
    with serial.Serial(port, baud, timeout=5.0) as ser:
        # Flush stale data
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # Wait for MCU to boot and print READY
        print("Waiting for MCU READY...", end="", flush=True)
        deadline = time.time() + 30
        while time.time() < deadline:
            line = ser.readline().decode("ascii", errors="replace").strip()
            if line:
                print(f"\r  MCU: {line}")
            if "READY" in line:
                break
        else:
            raise TimeoutError("MCU did not send READY within 30 s")

        # Send CSV rows
        n_rows = len(df)
        print(f"Sending {n_rows} rows...", end="", flush=True)
        chunk_size = 200
        for i, row in df.iterrows():
            line = f"{row['acc']:.8f},{row['acoustic']:.8f},"  \
                   f"{row['fx']:.8f},{row['fy']:.8f},{row['fz']:.8f}\n"
            ser.write(line.encode("ascii"))
            if (i + 1) % chunk_size == 0:
                # Brief pause to avoid overflowing the MCU UART RX buffer
                time.sleep(0.01)
                print(f"\r  Sent {i+1}/{n_rows} rows...", end="", flush=True)

        ser.write(b"END\n")
        ser.flush()
        print(f"\r  Sent {n_rows} rows + END")

        # Read MCU output until we see the prediction line
        print(f"Waiting for prediction (timeout {TIMEOUT_S} s)...")
        deadline = time.time() + TIMEOUT_S
        result = None
        while time.time() < deadline:
            line = ser.readline().decode("ascii", errors="replace").strip()
            if line:
                print(f"  MCU: {line}")
            if "Predicted tool wear" in line or "wear:" in line.lower():
                result = line
                break

        if result is None:
            raise TimeoutError("MCU did not return a prediction")
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Stream a cutting CSV to FRDM-MCXN947 and read wear prediction"
    )
    parser.add_argument("--port", required=True,
                        help="Serial port, e.g. /dev/tty.usbmodemXXXX  or COM3")
    parser.add_argument("--csv",  required=True,
                        help="Path to raw sensor CSV file")
    parser.add_argument("--baud", type=int, default=BAUD)
    parser.add_argument("--no-mask", action="store_true",
                        help="Skip cutting-mask extraction (send full CSV)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")

    if args.no_mask:
        raw = pd.read_csv(csv_path, header=None)
        raw.columns = ALL_COLS[:raw.shape[1]]
        df = raw[SENSOR_COLS]
    else:
        df = load_cutting_segment(csv_path)

    result = send_and_receive(args.port, args.baud, df)
    print(f"\n=== Result ===\n{result}")


if __name__ == "__main__":
    main()
