from dataclasses import dataclass, field

from sensors.sensor_frame import SensorFrame


@dataclass
class FrameState:
    timestamp: float
    frame_id: int
    phase: str
    confidence: float
    landmarks: dict
    metrics: dict
    sensor: SensorFrame
    detected_errors: list[str] = field(default_factory=list)
    realtime_feedback: str | None = None
    selected_error: str | None = None
    deferred_errors: list[str] = field(default_factory=list)
    decision_reason: str = ""
    total_reps: int = 0
    set_index: int = 1
    set_rep_count: int = 0
    rep_completed: bool = False
    calibration_status: str = "measuring"
    calibration_progress: float = 0.0
    calibration_message: str = ""
    calibration_baseline: dict = field(default_factory=dict)
