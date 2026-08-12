from __future__ import annotations

import math
import time


def quaternion_to_euler(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm < 0.5:
        raise ValueError("invalid quaternion")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    pitch = math.degrees(math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w*y - z*x)))))
    roll = math.degrees(math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)))
    return pitch, roll, yaw


class IMUReader:
    """Hardware-neutral IMU interface.

    Replace read_raw() with the installed IMU's datasheet driver. The rest of
    the project consumes the same quaternion/accel/gyro contract.
    """

    def __init__(self, mock: bool = False, spi: bool = True, cs_pin: str = "D8", int_pin: str = "D25", reset_pin: str = "D24"):
        self.mock = mock
        self.started = time.monotonic()
        self.last = (0.0, 0.0, 0.0)
        self.sensor = None
        if not mock:
            # BNO085 is the IMU used by the original Arduino sketch. The
            # Adafruit CircuitPython driver keeps the same quaternion,
            # acceleration and calibrated gyro data available on Pi SPI.
            import board
            import busio
            import digitalio
            from adafruit_bno08x import BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE, BNO_REPORT_ROTATION_VECTOR
            from adafruit_bno08x.spi import BNO08X_SPI
            cs = digitalio.DigitalInOut(getattr(board, cs_pin))
            int_line = digitalio.DigitalInOut(getattr(board, int_pin))
            reset = digitalio.DigitalInOut(getattr(board, reset_pin))
            self.sensor = BNO08X_SPI(busio.SPI(board.SCK, board.MOSI, board.MISO), cs, int_line, reset)
            self.sensor.enable_feature(BNO_REPORT_ACCELEROMETER)
            self.sensor.enable_feature(BNO_REPORT_GYROSCOPE)
            self.sensor.enable_feature(BNO_REPORT_ROTATION_VECTOR)

    def read_raw(self) -> dict[str, float]:
        if self.mock:
            t = time.monotonic() - self.started
            return {"acc_x": 0.0, "acc_y": 0.0, "acc_z": 9.81, "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0, "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0}
        if self.sensor is None:
            raise RuntimeError("BNO085 is not initialized")
        ax, ay, az = self.sensor.acceleration
        gx, gy, gz = self.sensor.gyro
        qw, qx, qy, qz = self.sensor.quaternion
        return {"acc_x": ax, "acc_y": ay, "acc_z": az, "gyro_x": gx, "gyro_y": gy, "gyro_z": gz, "qw": qw, "qx": qx, "qy": qy, "qz": qz}

    def read(self) -> dict[str, float]:
        raw = self.read_raw()
        try:
            pitch, roll, yaw = quaternion_to_euler(raw["qw"], raw["qx"], raw["qy"], raw["qz"])
        except (KeyError, ValueError):
            pitch = math.degrees(math.atan2(-raw["acc_x"], math.sqrt(raw["acc_y"]**2 + raw["acc_z"]**2)))
            roll = math.degrees(math.atan2(raw["acc_y"], raw["acc_z"]))
            yaw = 0.0
        gyro_rate = math.sqrt(raw["gyro_x"]**2 + raw["gyro_y"]**2 + raw["gyro_z"]**2)
        instability = max(0.0, min(1.0, gyro_rate / 8.0))
        self.last = (pitch, roll, yaw)
        return {"pitch": pitch, "roll": roll, "yaw": yaw, "instability": instability, **raw}
