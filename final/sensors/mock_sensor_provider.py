import math
import time

from .sensor_frame import SensorFrame


class MockSensorProvider:
    """Small deterministic mock for future FSR/IMU replacement."""

    def __init__(self) -> None:
        self.started_at = time.time()

    def read(self) -> SensorFrame:
        now = time.time()
        t = now - self.started_at
        left = 0.5 + 0.08 * math.sin(t * 0.7)
        right = 1.0 - left
        return SensorFrame(
            timestamp=now,
            left_pressure_ratio=round(left, 3),
            right_pressure_ratio=round(right, 3),
            heel_pressure_drop=math.sin(t * 0.33) > 0.92,
            forefoot_bias=math.sin(t * 0.29) > 0.9,
            imu_pitch=round(8.0 * math.sin(t * 0.5), 2),
            imu_roll=round(4.0 * math.sin(t * 0.9), 2),
            imu_instability=round(abs(math.sin(t * 1.3)) * 0.25, 3),
            core_pressure_ratio=round(0.72 + 0.08 * math.sin(t * 0.41), 3),
            source="mock",
        )
