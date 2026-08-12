from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from hardware.sensor_contract import SensorFrame


@dataclass
class Calibration:
    duration: float = 3.0
    started: float | None = None
    frames: list[SensorFrame] = field(default_factory=list)
    completed: bool = False
    baseline: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.started = time.time()
        self.frames.clear()
        self.completed = False
        self.baseline = {}

    def update(self, frame: SensorFrame) -> dict:
        if self.started is None:
            self.reset()
        if not self.completed:
            self.frames.append(frame)
            if time.time() - self.started >= self.duration:
                self.completed = True
                self.baseline = self._make_baseline()
        elapsed = max(0.0, time.time() - (self.started or time.time()))
        return {"status": "ready" if self.completed else "measuring", "progress": 1.0 if self.completed else min(1.0, elapsed / self.duration), "baseline": self.baseline, "samples": len(self.frames)}

    def _make_baseline(self) -> dict:
        def mean(values):
            values = [v for v in values if v is not None]
            return round(statistics.fmean(values), 4) if values else None
        return {"fsr": [mean([f.fsr[i] for f in self.frames]) for i in range(6)], "left_pressure_ratio": mean([f.left_pressure_ratio for f in self.frames]), "right_pressure_ratio": mean([f.right_pressure_ratio for f in self.frames]), "imu_pitch": mean([f.imu_pitch for f in self.frames]), "imu_roll": mean([f.imu_roll for f in self.frames]), "core_pressure_ratio": mean([f.core_pressure_ratio for f in self.frames])}
