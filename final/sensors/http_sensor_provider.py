import threading
import time
from typing import Any

from .sensor_frame import SensorFrame


class HttpSensorProvider:
    """Accept normalized sensor frames pushed through the browser backend.

    This is a bridge boundary, not raw Arduino acquisition. A hardware process
    can POST already-normalized SafeLift JSON to /api/sensor_frame; the main
    analysis loop reads the latest validated frame from this provider.
    """

    def __init__(self, stale_after_seconds: float = 2.0) -> None:
        self.stale_after_seconds = max(0.1, float(stale_after_seconds))
        self._lock = threading.Lock()
        self._latest: SensorFrame | None = None
        self._last_push_time = 0.0

    def push(self, payload: dict[str, Any]) -> SensorFrame:
        frame = SensorFrame.from_payload(payload, source="http")
        with self._lock:
            self._latest = frame
            self._last_push_time = time.time()
        return frame

    def read(self) -> SensorFrame:
        with self._lock:
            frame = self._latest
            age = time.time() - self._last_push_time if self._last_push_time else None
        if frame is None:
            return SensorFrame.invalid("http-no-frame", "no normalized sensor frame has been posted to /api/sensor_frame")
        if age is not None and age > self.stale_after_seconds:
            return SensorFrame.invalid("http-stale", f"latest /api/sensor_frame is stale after {age:.2f}s")
        return frame

    def close(self) -> None:
        pass
