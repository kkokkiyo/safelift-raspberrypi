from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SensorFrame:
    timestamp: float
    fsr: list[float] = field(default_factory=lambda: [0.0] * 6)
    left_pressure_ratio: float | None = None
    right_pressure_ratio: float | None = None
    imu_pitch: float | None = None
    imu_roll: float | None = None
    imu_yaw: float | None = None
    imu_instability: float | None = None
    core_pressure_ratio: float | None = None
    pressure_kpa: float | None = None
    source: str = "raspberry"
    valid: bool = True
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
