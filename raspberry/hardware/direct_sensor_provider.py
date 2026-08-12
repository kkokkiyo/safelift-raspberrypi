from __future__ import annotations

import time

from .adc_mcp3008 import MCP3008
from .fsr_reader import FSRReader
from .imu_reader import IMUReader
from .pressure_reader import PressureReader
from .sensor_contract import SensorFrame


class DirectSensorProvider:
    """Reads all sensors directly from Raspberry Pi peripherals."""

    def __init__(self, mock: bool = False):
        self.mock = mock
        self.init_errors: list[str] = []
        self.adc = MCP3008(mock=mock)
        self.fsr = FSRReader(self.adc)
        try:
            self.imu = IMUReader(mock=mock)
        except Exception as exc:
            self.imu = None
            self.init_errors.append(f"imu_init:{exc}")
        try:
            self.pressure = PressureReader(mock=mock)
        except Exception as exc:
            self.pressure = None
            self.init_errors.append(f"pressure_init:{exc}")
        self.pressure_baseline: float | None = None

    def read(self) -> SensorFrame:
        now = time.time()
        errors: list[str] = []
        try:
            fsr = self.fsr.read()
            left = sum(max(0.0, fsr[i]) for i in (0, 1, 2))
            right = sum(max(0.0, fsr[i]) for i in (3, 4, 5))
            total = left + right
            left_ratio = left / total if total > 1e-6 else None
            right_ratio = right / total if total > 1e-6 else None
        except Exception as exc:
            fsr, left_ratio, right_ratio = [0.0] * 6, None, None
            errors.append(f"fsr:{exc}")
        try:
            if self.imu is None:
                raise RuntimeError("IMU unavailable")
            imu = self.imu.read()
        except Exception as exc:
            imu = {"pitch": None, "roll": None, "yaw": None, "instability": None}
            errors.append(f"imu:{exc}")
        try:
            if self.pressure is None:
                raise RuntimeError("pressure sensor unavailable")
            pressure = self.pressure.read_kpa()
            if self.pressure_baseline is None and pressure > 0:
                self.pressure_baseline = pressure
            ratio = pressure / max(self.pressure_baseline or pressure, 1e-6)
        except Exception as exc:
            pressure, ratio = None, None
            errors.append(f"pressure:{exc}")
        errors = self.init_errors + errors
        return SensorFrame(now, fsr, left_ratio, right_ratio, imu.get("pitch"), imu.get("roll"), imu.get("yaw"), imu.get("instability"), ratio, pressure, valid=not errors, errors=errors)

    def close(self) -> None:
        self.adc.close()
        if self.pressure is not None:
            self.pressure.close()
