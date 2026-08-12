from __future__ import annotations

import time


class MCP3008:
    """Small MCP3008 SPI driver. Values are returned as 0..1023."""

    def __init__(self, bus: int = 0, device: int = 0, max_speed_hz: int = 1_000_000, mock: bool = False):
        self.mock = mock
        self._phase = 0.0
        self.spi = None
        if not mock:
            import spidev
            self.spi = spidev.SpiDev()
            self.spi.open(bus, device)
            self.spi.max_speed_hz = max_speed_hz
            self.spi.mode = 0

    def read(self, channel: int) -> int:
        if not 0 <= channel <= 7:
            raise ValueError("MCP3008 channel must be 0..7")
        if self.mock:
            self._phase += 0.03
            return max(0, min(1023, int(450 + 100 * __import__("math").sin(self._phase + channel))))
        response = self.spi.xfer2([1, (8 + channel) << 4, 0])
        return ((response[1] & 3) << 8) | response[2]

    def close(self) -> None:
        if self.spi is not None:
            self.spi.close()
