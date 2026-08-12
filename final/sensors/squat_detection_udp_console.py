"""Console monitor for the updated SquatDetection_Package UDP output.

The package's Arduino bridge owns the serial port and the FSR/Bracing
detectors publish normalized feature JSON to UDP port 24018. This monitor does
not open COM7, so it can run alongside the package without stealing the port.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from typing import Any


def fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}{suffix}"
    return f"{value}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Show updated SquatDetection sensor values")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=24018)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    sock.settimeout(0.1)
    print(f"[SquatDetection] listening for detector output on {args.host}:{args.port}", flush=True)
    print("[SquatDetection] COM7 is owned by the package bridge; press Ctrl+C to stop.", flush=True)

    latest_fsr: dict[str, Any] = {}
    latest_pressure: dict[str, Any] = {}
    packet_count = 0
    invalid_count = 0
    next_report = 0.0
    try:
        while True:
            try:
                raw, _ = sock.recvfrom(65535)
            except socket.timeout:
                raw = b""
            if raw:
                try:
                    packet = json.loads(raw.decode("utf-8", errors="replace"))
                    if not isinstance(packet, dict):
                        raise ValueError("packet is not an object")
                    packet_count += 1
                    if isinstance(packet.get("fsr"), dict) and packet["fsr"].get("valid") is not False:
                        latest_fsr = packet
                    if isinstance(packet.get("pressure"), dict):
                        latest_pressure = packet
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    invalid_count += 1

            now = time.monotonic()
            if now >= next_report:
                next_report = now + max(0.05, args.interval)
                fsr = latest_fsr.get("fsr") or {}
                pressure = (latest_pressure.get("pressure") or latest_fsr.get("pressure") or {})
                print(
                    "[SquatDetection] packets={packets} invalid={invalid} "
                    "FSR(L/R)={left}/{right} heelDrop={heel} "
                    "pressure={current}kPa bracingLoss={loss} status={status}".format(
                        packets=packet_count,
                        invalid=invalid_count,
                        left=fmt(fsr.get("leftPercent", fsr.get("leftRatio", fsr.get("left_ratio"))), "%"),
                        right=fmt(fsr.get("rightPercent", fsr.get("rightRatio", fsr.get("right_ratio"))), "%"),
                        heel=fsr.get("heelDroppedSide", fsr.get("heelDrop", fsr.get("heel_pressure_drop", "-"))),
                        current=fmt(pressure.get("current")),
                        loss=fmt(pressure.get("bracingLoss", pressure.get("dropRatio")), "%"),
                        status=pressure.get("status", fsr.get("status", "WAIT")),
                    ),
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\n[SquatDetection] stopped.", flush=True)
        return 0
    finally:
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
