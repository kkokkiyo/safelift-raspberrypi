"""Decode and normalize the SafeLift Arduino SLXR serial stream."""

from __future__ import annotations

import math
import time


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _quat_to_euler(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 0.5:
        raise ValueError("invalid quaternion")
    w, x, y, z = (w / norm, x / norm, y / norm, z / norm)
    pitch = math.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    sinp = _clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    yaw = math.degrees(math.asin(sinp))
    roll = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return pitch, roll, yaw


class ArduinoSlxrNormalizer:
    """Convert SLXR CSV to the normalized SafeLift sensor contract.

    The current firmware has six unlabeled FSR channels. Channels 1-3 are
    treated as the left foot and 4-6 as the right foot. Pressure is calibrated
    from the first standing samples.
    """

    def __init__(self, baseline_seconds: float = 3.0) -> None:
        self.started_at = time.monotonic()
        self.baseline_seconds = max(0.5, float(baseline_seconds))
        self.pressure_samples: list[float] = []
        self.last_pitch: float | None = None
        self.last_time: float | None = None

    def decode(self, line: str) -> dict:
        parts = line.strip().split(",")
        if len(parts) < 19 or parts[0] != "SLXR":
            raise ValueError("expected SLXR CSV with 18 sensor values")
        try:
            millis = float(parts[1])
            fsr = [float(value) for value in parts[2:8]]
            acc = [float(value) for value in parts[8:11]]
            gyro = [float(value) for value in parts[11:14]]
            quat = [float(value) for value in parts[14:18]]
            pressure_kpa = float(parts[18])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid SLXR numeric field: {exc}") from exc

        now = time.monotonic()
        if now - self.started_at <= self.baseline_seconds and pressure_kpa > 0:
            self.pressure_samples.append(pressure_kpa)
        baseline = sum(self.pressure_samples) / len(self.pressure_samples) if self.pressure_samples else pressure_kpa
        baseline = max(abs(baseline), 1e-6)

        left_raw = sum(max(0.0, value) for value in fsr[:3])
        right_raw = sum(max(0.0, value) for value in fsr[3:])
        total_raw = left_raw + right_raw
        left_ratio = left_raw / total_raw if total_raw > 1e-6 else None
        right_ratio = right_raw / total_raw if total_raw > 1e-6 else None

        try:
            pitch, roll, _ = _quat_to_euler(*quat)
        except ValueError:
            pitch = math.degrees(math.atan2(-acc[0], math.sqrt(acc[1] ** 2 + acc[2] ** 2)))
            roll = math.degrees(math.atan2(acc[1], acc[2]))

        previous_pitch = self.last_pitch
        previous_time = self.last_time
        self.last_pitch = pitch
        self.last_time = now
        pitch_rate = 0.0
        if previous_pitch is not None and previous_time is not None and now > previous_time:
            pitch_rate = abs(pitch - previous_pitch) / (now - previous_time)
        gyro_rate = math.sqrt(sum(value * value for value in gyro))

        return {
            "timestamp": millis / 1000.0,
            "left_pressure_ratio": left_ratio,
            "right_pressure_ratio": right_ratio,
            "heel_pressure_drop": None,
            "forefoot_bias": None,
            "forefoot_shift_percent": None,
            "fsr_scoring_type": "A",
            "imu_pitch": _clamp(pitch, -180.0, 180.0),
            "imu_roll": _clamp(roll, -180.0, 180.0),
            "imu_instability": _clamp((gyro_rate + pitch_rate * 0.02) / 8.0, 0.0, 1.0),
            "core_pressure_ratio": _clamp(pressure_kpa / baseline, 0.0, 1.0),
            "source": "arduino",
        }
