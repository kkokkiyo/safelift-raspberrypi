import json
from collections import Counter
from pathlib import Path

from .frame_state import FrameState


class RepSummaryBuilder:
    def build(self, rep_id: int, frame_states: list[FrameState], phase_snapshots: dict) -> dict:
        if not frame_states:
            return {"rep_id": rep_id, "phase_snapshots": phase_snapshots, "rule_analysis": {}, "metrics": {}, "phase_notes": {}}

        all_errors = [error for state in frame_states for error in state.detected_errors]
        selected_feedback = [state.realtime_feedback for state in frame_states if state.realtime_feedback]
        deferred_errors = sorted({error for state in frame_states for error in state.deferred_errors})
        metrics = self._aggregate_metrics(frame_states)
        main_issue = Counter(all_errors).most_common(1)[0][0] if all_errors else None

        return {
            "rep_id": rep_id,
            "start_time": frame_states[0].timestamp,
            "end_time": frame_states[-1].timestamp,
            "duration_seconds": round(frame_states[-1].timestamp - frame_states[0].timestamp, 2),
            "phase_snapshots": phase_snapshots,
            "rule_analysis": {
                "main_issue": main_issue,
                "detected_errors": sorted(set(all_errors)),
                "selected_realtime_feedback": list(dict.fromkeys(selected_feedback)),
                "deferred_errors": deferred_errors,
            },
            "metrics": metrics,
            "phase_notes": self._phase_notes(frame_states),
        }

    def save(self, rep_summary: dict, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rep_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def _aggregate_metrics(self, states: list[FrameState]) -> dict:
        def values(key: str) -> list[float]:
            return [float(state.metrics[key]) for state in states if state.metrics.get(key) is not None]

        def sensor_values(key: str) -> list[float]:
            out = []
            for state in states:
                value = getattr(state.sensor, key)
                if value is not None:
                    out.append(float(value))
            return out

        torso = values("torso_angle")
        side_torso = values("side_torso_angle")
        knees = values("avg_knee_angle")
        valgus = values("knee_valgus_score")
        left_pressure = sensor_values("left_pressure_ratio")
        right_pressure = sensor_values("right_pressure_ratio")
        imu_pitch = sensor_values("imu_pitch")
        imu_roll = sensor_values("imu_roll")
        core_pressure = sensor_values("core_pressure_ratio")
        forefoot_shift = sensor_values("forefoot_shift_percent")
        fsr_types = [state.sensor.fsr_scoring_type for state in states if state.sensor.fsr_scoring_type in {"A", "B"}]

        return {
            "max_torso_angle": round(max(torso), 2) if torso else None,
            "max_side_torso_angle": round(max(side_torso), 2) if side_torso else None,
            "min_knee_angle": round(min(knees), 2) if knees else None,
            "max_knee_valgus_score": round(max(valgus), 4) if valgus else None,
            "left_pressure_peak_ratio": round(max(left_pressure), 3) if left_pressure else None,
            "right_pressure_min_ratio": round(min(right_pressure), 3) if right_pressure else None,
            "heel_pressure_drop": any(state.sensor.heel_pressure_drop is True for state in states),
            "forefoot_shift_peak_percent": round(max(forefoot_shift), 2) if forefoot_shift else None,
            "fsr_scoring_type": max(set(fsr_types), key=fsr_types.count) if fsr_types else "A",
            "imu_pitch_peak": round(max(imu_pitch), 2) if imu_pitch else None,
            "imu_roll_peak": round(max(imu_roll, key=abs), 2) if imu_roll else None,
            "core_pressure_min_ratio": round(min(core_pressure), 3) if core_pressure else None,
        }

    def _phase_notes(self, states: list[FrameState]) -> dict:
        notes = {}
        for phase in ["Standing", "Descent", "Bottom", "Ascent"]:
            phase_states = [state for state in states if state.phase == phase]
            if not phase_states:
                continue
            errors = Counter(error for state in phase_states for error in state.detected_errors)
            if errors:
                notes[phase.lower()] = f"{phase} 구간에서 {errors.most_common(1)[0][0]} 후보가 가장 자주 감지됨"
            else:
                notes[phase.lower()] = f"{phase} 구간에서 큰 rule-based 오류는 감지되지 않음"
        return notes
