from __future__ import annotations

from dataclasses import dataclass, field

from sensors.sensor_frame import SensorFrame


@dataclass
class CalibrationTracker:
    """Tracks the tutorial standing-baseline window.

    This class does not acquire raw sensor data. It summarizes whatever sensor
    frame and pose confidence the active provider already produced, so Unity
    can show a real calibration status instead of a static placeholder.
    """

    duration_seconds: float = 3.0
    started_at: float | None = None
    completed: bool = False
    sensor_samples: list[SensorFrame] = field(default_factory=list)
    confidence_samples: list[float] = field(default_factory=list)

    def update(self, now: float, confidence: float, sensor: SensorFrame) -> dict:
        if self.started_at is None:
            self.started_at = now

        if not self.completed:
            self.sensor_samples.append(sensor)
            self.confidence_samples.append(float(confidence or 0.0))
            if now - self.started_at >= self.duration_seconds:
                self.completed = True

        return self.to_dict(now)

    def reset(self) -> None:
        self.started_at = None
        self.completed = False
        self.sensor_samples.clear()
        self.confidence_samples.clear()

    def to_dict(self, now: float) -> dict:
        elapsed = 0.0 if self.started_at is None else max(0.0, now - self.started_at)
        progress = 1.0 if self.completed else min(1.0, elapsed / max(0.01, self.duration_seconds))
        confidence_mean = _mean(self.confidence_samples)
        source = _most_common([sample.source or "none" for sample in self.sensor_samples]) or "none"

        status = "ready" if self.completed else "measuring"
        if source == "none":
            sensor_message = "No physical sensor provider; using pose-only/mock baseline."
        elif source == "mock":
            sensor_message = "Mock sensor baseline is active."
        else:
            sensor_message = f"{source} sensor baseline is active."

        if confidence_mean is not None and confidence_mean < 0.25:
            status = "poor_tracking" if not self.completed else "ready_with_low_tracking"
            tracking_message = "Move so the full body is visible to the camera."
        else:
            tracking_message = "Tracking baseline is stable."

        return {
            "status": status,
            "progress": round(progress, 3),
            "duration_seconds": self.duration_seconds,
            "elapsed_seconds": round(elapsed, 3),
            "message": f"{sensor_message} {tracking_message}",
            "baseline": {
                "sensor_mode": source,
                "sample_count": len(self.sensor_samples),
                "pose_confidence_mean": round(confidence_mean, 3) if confidence_mean is not None else None,
                "left_pressure_ratio": _stats([s.left_pressure_ratio for s in self.sensor_samples]),
                "right_pressure_ratio": _stats([s.right_pressure_ratio for s in self.sensor_samples]),
                "core_pressure_ratio": _stats([getattr(s, "core_pressure_ratio", None) for s in self.sensor_samples]),
                "imu_pitch": _stats([s.imu_pitch for s in self.sensor_samples]),
                "imu_roll": _stats([s.imu_roll for s in self.sensor_samples]),
            },
        }


def _mean(values: list[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _stats(values: list[float | None]) -> dict:
    clean = [float(value) for value in values if value is not None]
    return {
        "mean": round(sum(clean) / len(clean), 3) if clean else None,
        "min": round(min(clean), 3) if clean else None,
        "max": round(max(clean), 3) if clean else None,
        "samples": len(clean),
    }


def _most_common(values: list[str]) -> str | None:
    if not values:
        return None
    return max(set(values), key=values.count)
