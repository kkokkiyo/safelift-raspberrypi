from __future__ import annotations

from collections.abc import Mapping
from statistics import mean

from .frame_state import FrameState


LOWER_BODY_LANDMARKS = {
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

REQUIRED_PHASES = ("standing", "descent", "bottom", "ascent")
CORE_REPORT_PHASES = ("bottom", "ascent")


class LastRepQualityEvaluator:
    """Evaluate whether a completed rep has enough evidence for a posture report."""

    def __init__(
        self,
        bad_threshold: float = 0.15,
        weak_threshold: float = 0.45,
        min_phase_frames: int = 1,
        require_core_captures: bool = False,
        allow_missing_phases: bool = True,
    ) -> None:
        self.bad_threshold = bad_threshold
        self.weak_threshold = weak_threshold
        self.min_phase_frames = min_phase_frames
        self.require_core_captures = require_core_captures
        self.allow_missing_phases = allow_missing_phases

    def evaluate(self, frame_states: list[FrameState], phase_captures: Mapping | None = None) -> dict:
        phase_captures = phase_captures or {}
        phase_quality = {
            phase: self._evaluate_phase(phase, frame_states, phase_captures)
            for phase in REQUIRED_PHASES
        }

        all_confidences = [float(state.confidence) for state in frame_states]
        all_visibility = [
            visibility
            for state in frame_states
            for visibility in self._lower_body_visibility_values(state)
        ]
        mean_confidence = mean(all_confidences) if all_confidences else 0.0
        mean_lower_visibility = mean(all_visibility) if all_visibility else 0.0

        core_levels = [phase_quality[phase]["level"] for phase in CORE_REPORT_PHASES]
        core_usable = all(level in {"good", "weak"} for level in core_levels)
        core_bad_count = sum(1 for level in core_levels if level == "bad")

        any_core_seen = any(
            phase_quality[phase]["frame_count"] > 0 or phase_quality[phase]["has_capture"]
            for phase in CORE_REPORT_PHASES
        )

        if not any_core_seen and mean_confidence < self.bad_threshold:
            level = "bad"
            reason = "bottom/ascent phase evidence is missing and overall tracking confidence is too low."
        elif core_bad_count >= 2 and self.require_core_captures:
            level = "bad"
            reason = "bottom/ascent phase evidence is mostly bad, so posture report is blocked."
        elif core_bad_count >= 2:
            level = "weak"
            reason = "bottom/ascent evidence is limited, but prototype mode allows a cautious report."
        elif any_core_seen and not self.require_core_captures:
            level = "weak"
            reason = "one core phase was captured; prototype mode allows a cautious report."
        elif core_usable:
            if (
                all(level == "good" for level in core_levels)
                and mean_confidence >= self.weak_threshold
                and mean_lower_visibility >= self.weak_threshold
            ):
                level = "good"
                reason = "bottom/ascent phase captures and rep tracking are reliable."
            else:
                level = "weak"
                reason = "bottom/ascent are usable, but confidence or lower-body visibility is limited."
        else:
            level = "bad"
            reason = "bottom or ascent phase evidence is not reliable enough for posture analysis."

        return {
            "level": level,
            "report_allowed": level in {"good", "weak"},
            "mean_confidence": round(mean_confidence, 3),
            "mean_lower_body_visibility": round(mean_lower_visibility, 3),
            "phase_quality": phase_quality,
            "reason": reason,
        }

    def _evaluate_phase(self, phase: str, frame_states: list[FrameState], phase_captures: Mapping) -> dict:
        phase_states = [state for state in frame_states if str(state.phase).lower() == phase]
        has_capture = self._has_phase_capture(phase, phase_captures)

        if not phase_states:
            if self.allow_missing_phases:
                return self._phase_result(
                    level="weak",
                    frame_count=0,
                    has_capture=has_capture,
                    mean_confidence=0.0,
                    mean_lower_visibility=0.0,
                    reason=f"{phase} phase has no tracked frames; prototype mode keeps the rep usable.",
                )
            return self._phase_result(
                level="bad",
                frame_count=0,
                has_capture=has_capture,
                mean_confidence=0.0,
                mean_lower_visibility=0.0,
                reason=f"{phase} phase has no tracked frames.",
            )

        confidences = [float(state.confidence) for state in phase_states]
        visibilities = [
            visibility
            for state in phase_states
            for visibility in self._lower_body_visibility_values(state)
        ]
        mean_confidence = mean(confidences) if confidences else 0.0
        mean_lower_visibility = mean(visibilities) if visibilities else 0.0

        if phase in CORE_REPORT_PHASES and not has_capture and self.require_core_captures:
            level = "bad"
            reason = f"{phase} phase capture is missing."
        elif mean_confidence < self.bad_threshold or mean_lower_visibility < self.bad_threshold:
            level = "bad"
            reason = f"{phase} confidence or lower-body visibility is below {self.bad_threshold:.2f}."
        elif len(phase_states) < self.min_phase_frames:
            level = "weak"
            reason = f"{phase} phase has only {len(phase_states)} tracked frame(s)."
        elif mean_confidence < self.weak_threshold or mean_lower_visibility < self.weak_threshold:
            level = "weak"
            reason = f"{phase} is usable but confidence or lower-body visibility is below {self.weak_threshold:.2f}."
        else:
            level = "good"
            reason = f"{phase} tracking quality is stable."

        return self._phase_result(
            level=level,
            frame_count=len(phase_states),
            has_capture=has_capture,
            mean_confidence=mean_confidence,
            mean_lower_visibility=mean_lower_visibility,
            reason=reason,
        )

    def _lower_body_visibility_values(self, state: FrameState) -> list[float]:
        values = []
        for index in LOWER_BODY_LANDMARKS:
            landmark = state.landmarks.get(index)
            values.append(float(landmark.get("visibility", 0.0)) if landmark else 0.0)
        return values

    def _has_phase_capture(self, phase: str, phase_captures: Mapping) -> bool:
        for key, value in phase_captures.items():
            if str(key).lower() == phase and value:
                return True
        return False

    def _phase_result(
        self,
        level: str,
        frame_count: int,
        has_capture: bool,
        mean_confidence: float,
        mean_lower_visibility: float,
        reason: str,
    ) -> dict:
        return {
            "level": level,
            "frame_count": frame_count,
            "has_capture": has_capture,
            "mean_confidence": round(mean_confidence, 3),
            "mean_lower_body_visibility": round(mean_lower_visibility, 3),
            "reason": reason,
        }
