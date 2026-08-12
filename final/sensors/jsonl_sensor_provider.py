import json
from pathlib import Path

from .sensor_frame import SensorFrame


class JsonlSensorProvider:
    """Replay already-normalized SafeLift sensor frames from a JSONL file.

    This is intentionally not a raw hardware reader. It lets the hardware team,
    tests, or a bridge process feed the same normalized contract that the serial
    provider expects, one JSON object per line. By default the file loops so
    Real Mode has a stable provider during demos and preflight checks.
    """

    def __init__(self, path: str, loop: bool = True) -> None:
        self.path = Path(path)
        self.loop = loop
        if not self.path.exists():
            raise FileNotFoundError(f"sensor JSONL file not found: {self.path}")
        self._lines = self.path.read_text(encoding="utf-8").splitlines()
        self._index = 0
        if not any(line.strip() for line in self._lines):
            raise ValueError(f"sensor JSONL file has no frames: {self.path}")

    def read(self) -> SensorFrame:
        if self._index >= len(self._lines):
            if not self.loop:
                return SensorFrame.invalid("jsonl-eof", "sensor JSONL reached end of file")
            self._index = 0

        line = self._lines[self._index].strip()
        self._index += 1
        if not line:
            return SensorFrame.invalid("jsonl-empty", "sensor JSONL line was empty")
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return SensorFrame.invalid("jsonl-invalid", "sensor JSONL line was not valid JSON")
        if not isinstance(data, dict):
            return SensorFrame.invalid("jsonl-invalid", "sensor JSONL frame must be an object")
        return SensorFrame.from_payload(data, source="jsonl")

    def close(self) -> None:
        pass
