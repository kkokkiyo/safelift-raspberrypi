from .frame_state import FrameState


class RuleErrorDetector:
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        torso_lean_threshold: float = 30.0,
        shallow_depth_angle: float = 120.0,
        valgus_threshold: float = 0.08,
        pressure_asymmetry_threshold: float = 0.2,
        imu_roll_threshold: float = 8.0,
        imu_instability_threshold: float = 0.45,
        core_pressure_drop_threshold: float = 0.55,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.torso_lean_threshold = torso_lean_threshold
        self.shallow_depth_angle = shallow_depth_angle
        self.valgus_threshold = valgus_threshold
        self.pressure_asymmetry_threshold = pressure_asymmetry_threshold
        self.imu_roll_threshold = imu_roll_threshold
        self.imu_instability_threshold = imu_instability_threshold
        self.core_pressure_drop_threshold = core_pressure_drop_threshold

    def detect(self, frame_state: FrameState) -> list[str]:
        errors: list[str] = []
        metrics = frame_state.metrics
        sensor = frame_state.sensor

        if frame_state.confidence < self.confidence_threshold:
            return ["TrackingUncertain"]
        if _gt(metrics.get("torso_angle"), self.torso_lean_threshold):
            errors.append("TorsoLean")
        if frame_state.phase == "Bottom" and _gt(metrics.get("avg_knee_angle"), self.shallow_depth_angle):
            errors.append("ShallowDepth")
        if _gt(metrics.get("knee_valgus_score"), self.valgus_threshold):
            errors.append("KneeValgus")

        if sensor.left_pressure_ratio is not None and sensor.right_pressure_ratio is not None:
            if abs(sensor.left_pressure_ratio - sensor.right_pressure_ratio) > self.pressure_asymmetry_threshold:
                errors.append("WeightAsymmetry")
        if sensor.heel_pressure_drop is True:
            errors.append("HeelLift")
        if sensor.core_pressure_ratio is not None and sensor.core_pressure_ratio < self.core_pressure_drop_threshold:
            errors.append("CorePressureDrop")
        if sensor.imu_roll is not None and abs(sensor.imu_roll) > self.imu_roll_threshold:
            errors.append("TorsoInstability")
        elif sensor.imu_instability is not None and sensor.imu_instability > self.imu_instability_threshold:
            errors.append("TorsoInstability")

        return errors


def _gt(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold
