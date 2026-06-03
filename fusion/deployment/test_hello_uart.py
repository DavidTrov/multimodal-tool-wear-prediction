#!/usr/bin/env python3
"""Minimal UART round-trip test for the FRDM-MCXN947 hello diagnostic firmware.

Waits for the board's "READY", sends "hello", and checks that the board echoes
it back. Proves the boot + UART path works before bringing the model back.

Usage:
    python fusion/deployment/test_hello_uart.py --port /dev/tty.usbmodemXXXX

Tip: list candidate ports with
    python -m serial.tools.list_ports
"""
import argparse
import sys
import time

import serial


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="serial device, e.g. /dev/tty.usbmodemXXXX")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout", type=float, default=20.0, help="seconds to wait for READY")
    args = ap.parse_args()

    print(f"Opening {args.port} @ {args.baud}...", flush=True)
    with serial.Serial(args.port, args.baud, timeout=1.0) as ser:
        
        print("Waiting for READY (press the board RESET button if nothing appears)...", flush=True)
        deadline = time.time() + args.timeout
        
        

        # Now it is safe to send hello
        print("  > hello", flush=True)
        ser.write(b"hello\n")
        ser.flush()

        deadline = time.time() + 5.0
        saw_echo = False
        saw_hello_back = False
        while time.time() < deadline:
            line = ser.readline().decode("ascii", errors="replace").strip()
            if not line:
                continue
            print(f"  < {line}")
            if line == "ECHO: hello":
                saw_echo = True
            if line == "hello back":
                saw_hello_back = True
            if saw_echo and saw_hello_back:
                break

        if saw_echo and saw_hello_back:
            print("\nPASS: board received 'hello' and replied correctly.")
            return 0
        print("\nFAIL: did not get the expected echo / reply.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
