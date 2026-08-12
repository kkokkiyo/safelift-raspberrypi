from __future__ import annotations

import time
from collections import deque

from .adc_mcp3008 import MCP3008
from .sensor_contract import SensorFrame


class FSRReader:
    def __init__(self, adc: MCP3008, channels: list[int] | None = None, window: int = 5):
        self.adc = adc
        self.channels = channels or [0, 1, 2, 3, 4, 5]
        self.history = [deque(maxlen=max(1, window)) for _ in self.channels]
        self.baseline = [0.0] * len(self.channels)

    def read_raw(self) -> list[float]:
        values = []
        for i, channel in enumerate(self.channels):
            value = float(self.adc.read(channel))
            self.history[i].append(value)
            values.append(sum(self.history[i]) / len(self.history[i]))
        return values

    def calibrate(self, samples: list[list[float]]) -> None:
        if samples:
            self.baseline = [sum(row[i] for row in samples) / len(samples) for i in range(len(self.channels))]

    def read(self) -> list[float]:
        raw = self.read_raw()
        # Do not subtract baseline: pressure magnitude is needed for balance.
        # Baseline is retained for diagnostics and future sensor-specific curves.
        return raw
