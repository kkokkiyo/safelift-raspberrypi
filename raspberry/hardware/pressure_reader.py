from __future__ import annotations


class PressureReader:
    """Hardware-neutral I2C pressure interface.

    The existing Arduino used SparkFun SEN-16476. Implement the same register
    sequence for the exact sensor board used on Raspberry Pi here.
    """

    def __init__(self, bus: int = 1, address: int = 0x18, mock: bool = False):
        self.mock = mock
        self.baseline = None
        self.bus = None
        if not mock:
            from smbus2 import SMBus
            self.bus = SMBus(bus)
            self.address = address

    def read_kpa(self) -> float:
        if self.mock:
            return 101.3
        # SEN-16476 is based on the MPRLS pressure sensor. It accepts the
        # 0xAA/0x00/0x00 measurement command and returns status + 24-bit raw.
        self.bus.write_i2c_block_data(self.address, 0xAA, [0x00, 0x00])
        import time
        time.sleep(0.005)
        data = self.bus.read_i2c_block_data(self.address, 0x00, 4)
        raw = (data[1] << 16) | (data[2] << 8) | data[3]
        status = data[0] & 0x20
        if status:
            raise RuntimeError(f"MPRLS status error: 0x{data[0]:02x}")
        # MPRLS0025PA is 0..25 psi with a 10..90% output span.
        pressure_psi = ((raw - 0.1 * 0xFFFFFF) / (0.8 * 0xFFFFFF)) * 25.0
        return max(0.0, pressure_psi * 6.894757)

    def close(self) -> None:
        if self.bus is not None:
            self.bus.close()
