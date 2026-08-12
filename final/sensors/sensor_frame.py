import math
import time
from dataclasses import dataclass, field
from typing import Any


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _ratio_field(payload: dict[str, Any], key: str, errors: list[str]) -> float | None:
    if key not in payload or payload.get(key) is None:
        return None
    value = _finite_float(payload.get(key))
    if value is None or value < 0.0 or value > 1.0:
        errors.append(f"{key} must be a normalized ratio from 0.0 to 1.0")
        return None
    return value


def _angle_field(payload: dict[str, Any], key: str, errors: list[str]) -> float | None:
    if key not in payload or payload.get(key) is None:
        return None
    value = _finite_float(payload.get(key))
    if value is None or abs(value) > 180.0:
        errors.append(f"{key} must be a finite angle in degrees within -180.0 to 180.0")
        return None
    return value


def _percent_field(payload: dict[str, Any], key: str, errors: list[str]) -> float | None:
    if key not in payload or payload.get(key) is None:
        return None
    value = _finite_float(payload.get(key))
    if value is None or value < 0.0 or value > 500.0:
        errors.append(f"{key} must be a finite percent from 0.0 to 500.0")
        return None
    return value


def _fsr_scoring_type(payload: dict[str, Any], errors: list[str]) -> str | None:
    if "fsr_scoring_type" not in payload or payload.get("fsr_scoring_type") is None:
        return None
    mode = str(payload.get("fsr_scoring_type")).strip().upper()
    if mode not in {"A", "B"}:
        errors.append("fsr_scoring_type must be either A or B")
        return None
    return mode


@dataclass
class SensorFrame:
    timestamp: float
    left_pressure_ratio: float | None = None
    right_pressure_ratio: float | None = None
    heel_pressure_drop: bool | None = None
    forefoot_bias: bool | None = None
    forefoot_shift_percent: float | None = None
    fsr_scoring_type: str | None = None
    imu_pitch: float | None = None
    imu_roll: float | None = None
    imu_instability: float | None = None
    core_pressure_ratio: float | None = None
    source: str = "mock"
    contract_ok: bool = True
    contract_errors: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], source: str = "serial") -> "SensorFrame":
        """Build a frame from an already-acquired normalized sensor payload.

        This intentionally does not read hardware. It validates the contract
        expected by SafeLift XR after a serial/provider layer has acquired raw
        values and normalized them for the backend.
        """

        errors: list[str] = []
        timestamp = _finite_float(payload.get("timestamp")) or time.time()
        imu_instability = _ratio_field(payload, "imu_instability", errors)
        left = _ratio_field(payload, "left_pressure_ratio", errors)
        right = _ratio_field(payload, "right_pressure_ratio", errors)
        core = _ratio_field(payload, "core_pressure_ratio", errors)
        pitch = _angle_field(payload, "imu_pitch", errors)
        roll = _angle_field(payload, "imu_roll", errors)
        forefoot_shift = _percent_field(payload, "forefoot_shift_percent", errors)
        fsr_type = _fsr_scoring_type(payload, errors)

        heel_drop = _optional_bool(payload.get("heel_pressure_drop"))
        if "heel_pressure_drop" in payload and payload.get("heel_pressure_drop") is not None and heel_drop is None:
            errors.append("heel_pressure_drop must be boolean-compatible")
        forefoot = _optional_bool(payload.get("forefoot_bias"))
        if "forefoot_bias" in payload and payload.get("forefoot_bias") is not None and forefoot is None:
            errors.append("forefoot_bias must be boolean-compatible")

        contract_ok = not errors
        return cls(
            timestamp=timestamp,
            left_pressure_ratio=left,
            right_pressure_ratio=right,
            heel_pressure_drop=heel_drop,
            forefoot_bias=forefoot,
            forefoot_shift_percent=forefoot_shift,
            fsr_scoring_type=fsr_type,
            imu_pitch=pitch,
            imu_roll=roll,
            imu_instability=imu_instability,
            core_pressure_ratio=core,
            source=source if contract_ok else f"{source}-contract-invalid",
            contract_ok=contract_ok,
            contract_errors=tuple(errors),
        )

    @classmethod
    def invalid(cls, source: str, message: str) -> "SensorFrame":
        return cls(
            timestamp=time.time(),
            source=source,
            contract_ok=False,
            contract_errors=(message,),
        )
