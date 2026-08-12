import json

from .arduino_slxr import ArduinoSlxrNormalizer
from .sensor_frame import SensorFrame


class SerialSensorProvider:
    """Reads newline-delimited JSON sensor frames from a serial port.

    Raw hardware acquisition and calibration are expected to happen before this
    contract boundary. This provider only accepts normalized SafeLift payloads
    and marks malformed/out-of-range frames as not ready for Real Mode.
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.01) -> None:
        import serial

        self.serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self.arduino = ArduinoSlxrNormalizer()

    def read(self) -> SensorFrame:
        line = self.serial.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            return SensorFrame.invalid("serial-empty", "serial line was empty")
        if line.startswith("SLXR,"):
            try:
                return SensorFrame.from_payload(self.arduino.decode(line), source="arduino")
            except ValueError as exc:
                return SensorFrame.invalid("serial-slxr-invalid", str(exc))
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return SensorFrame.invalid("serial-invalid", "serial line was not valid JSON")
        if not isinstance(data, dict):
            return SensorFrame.invalid("serial-invalid", "serial JSON frame must be an object")
        return SensorFrame.from_payload(data, source="serial")

    def close(self) -> None:
        self.serial.close()
