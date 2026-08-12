from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


ISSUE_META: dict[str, dict[str, Any]] = {
    "KneeValgus": {
        "body_area": "knee",
        "severity": "warning",
        "haptic_channel": "both",
        "display_priority": 90,
        "default_phase": "Ascent",
        "cue": "Push both knees outward so they track over your toes.",
    },
    "WeightShiftLeft": {
        "body_area": "balance",
        "severity": "warning",
        "haptic_channel": "left",
        "display_priority": 82,
        "default_phase": "Bottom",
        "cue": "Shift pressure back toward the center of both feet.",
    },
    "WeightShiftRight": {
        "body_area": "balance",
        "severity": "warning",
        "haptic_channel": "right",
        "display_priority": 82,
        "default_phase": "Bottom",
        "cue": "Shift pressure back toward the center of both feet.",
    },
    "WeightAsymmetry": {
        "body_area": "balance",
        "severity": "warning",
        "haptic_channel": "both",
        "display_priority": 80,
        "default_phase": "Bottom",
        "cue": "Even out your left and right foot pressure.",
    },
    "CorePressureDrop": {
        "body_area": "core",
        "severity": "warning",
        "haptic_channel": "double",
        "display_priority": 78,
        "default_phase": "Descent",
        "cue": "Re-brace your core before the next ascent.",
    },
    "TorsoLean": {
        "body_area": "torso",
        "severity": "warning",
        "haptic_channel": "both",
        "display_priority": 74,
        "default_phase": "Descent",
        "cue": "Keep your chest tall and control the torso angle.",
    },
    "TorsoInstability": {
        "body_area": "torso",
        "severity": "warning",
        "haptic_channel": "both",
        "display_priority": 70,
        "default_phase": "Ascent",
        "cue": "Slow down and stabilize your torso.",
    },
    "ShallowDepth": {
        "body_area": "depth",
        "severity": "info",
        "haptic_channel": "both",
        "display_priority": 66,
        "default_phase": "Bottom",
        "cue": "If comfortable, lower slightly deeper before standing.",
    },
    "HeelLift": {
        "body_area": "foot",
        "severity": "warning",
        "haptic_channel": "both",
        "display_priority": 64,
        "default_phase": "Descent",
        "cue": "Keep your heels grounded through the full foot.",
    },
    "TrackingUncertain": {
        "body_area": "tracking",
        "severity": "info",
        "haptic_channel": "warning",
        "display_priority": 30,
        "default_phase": "Unknown",
        "cue": "Pose tracking is uncertain. Adjust camera or body position.",
    },
    "None": {
        "body_area": "none",
        "severity": "none",
        "haptic_channel": "none",
        "display_priority": 0,
        "default_phase": "Unknown",
        "cue": "",
    },
}


def _sensor_dict(sensor: Any) -> dict[str, Any]:
    if sensor is None:
        return {}
    if isinstance(sensor, dict):
        return sensor
    if is_dataclass(sensor):
        return asdict(sensor)
    return {
        "left_pressure_ratio": getattr(sensor, "left_pressure_ratio", None),
        "right_pressure_ratio": getattr(sensor, "right_pressure_ratio", None),
        "core_pressure_ratio": getattr(sensor, "core_pressure_ratio", None),
    }


def canonical_issue(issue: str | None, sensor: Any = None) -> str:
    value = str(issue or "None")
    if value in {"", "null", "None"}:
        return "None"
    if value == "WeightShift":
        value = "WeightAsymmetry"
    if value == "WeightAsymmetry":
        data = _sensor_dict(sensor)
        left = data.get("left_pressure_ratio")
        right = data.get("right_pressure_ratio")
        try:
            if left is not None and right is not None:
                return "WeightShiftLeft" if float(left) > float(right) else "WeightShiftRight"
        except (TypeError, ValueError):
            pass
    return value


def describe_live_feedback(
    issue: str | None,
    phase: str | None,
    cue: str | None = None,
    sensor: Any = None,
    reason: str | None = None,
) -> dict[str, Any]:
    key = canonical_issue(issue, sensor)
    meta = dict(ISSUE_META.get(key) or ISSUE_META.get(str(issue or ""), ISSUE_META["None"]))
    resolved_cue = cue or meta.get("cue") or ""
    return {
        "issue": key,
        "body_area": meta["body_area"],
        "phase": phase or meta.get("default_phase") or "Unknown",
        "severity": meta["severity"],
        "haptic_channel": meta["haptic_channel"],
        "display_priority": int(meta["display_priority"]),
        "cue": resolved_cue,
        "reason": reason or "",
    }
