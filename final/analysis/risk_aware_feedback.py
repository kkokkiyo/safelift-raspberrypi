from .live_feedback_contract import describe_live_feedback

class RiskAwareFeedbackEngine:
    PRIORITY = {
        "Standing": ["WeightAsymmetry", "TorsoInstability", "TrackingUncertain"],
        "Descent": ["TorsoLean", "HeelLift", "WeightAsymmetry", "KneeValgus"],
        "Bottom": ["HeelLift", "ShallowDepth", "TorsoLean", "WeightAsymmetry"],
        "Ascent": ["KneeValgus", "WeightAsymmetry", "TorsoInstability", "TorsoLean"],
        "Unknown": ["TrackingUncertain", "TorsoLean", "WeightAsymmetry", "KneeValgus"],
    }

    FEEDBACK = {
        "KneeValgus": "무릎을 발끝 방향으로 밀어주세요.",
        "TorsoLean": "가슴을 세우고 상체를 안정적으로 유지하세요.",
        "ShallowDepth": "가능하면 조금 더 깊게 내려가세요.",
        "WeightAsymmetry": "양발에 체중을 균등하게 실어주세요.",
        "HeelLift": "뒤꿈치를 바닥에 붙여주세요.",
        "TorsoInstability": "몸통이 흔들립니다. 천천히 움직여주세요.",
        "TrackingUncertain": "자세 추적이 불안정합니다. 카메라 위치를 조정해주세요.",
    }

    def __init__(self, cooldown_seconds: float = 2.0, max_cues_per_rep: int = 3, max_cues_per_phase: int = 2) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.max_cues_per_rep = max_cues_per_rep
        self.max_cues_per_phase = max_cues_per_phase
        self.last_error_time: dict[str, float] = {}
        self.phase_counts: dict[str, int] = {}
        self.rep_cue_count = 0

    def reset_rep(self) -> None:
        self.phase_counts.clear()
        self.rep_cue_count = 0

    def select_feedback(self, phase: str, errors: list[str], timestamp: float) -> dict:
        if not errors:
            return {"selected_error": None, "feedback": None, "deferred_errors": [], "decision_reason": "No rule error detected.", "live_feedback": describe_live_feedback(None, phase)}
        priority = self.PRIORITY.get(phase, self.PRIORITY["Unknown"])
        ordered = [error for error in priority if error in errors] + [error for error in errors if error not in priority]
        deferred = ordered[1:]
        phase_count = self.phase_counts.get(phase, 0)

        for error in ordered:
            last = self.last_error_time.get(error, -9999.0)
            if timestamp - last < self.cooldown_seconds:
                continue
            if phase_count >= self.max_cues_per_phase:
                return {
                    "selected_error": None,
                    "feedback": None,
                    "deferred_errors": ordered,
                    "decision_reason": f"{phase} phase cue limit reached.",
                    "live_feedback": describe_live_feedback(None, phase, reason=f"{phase} phase cue limit reached."),
                }
            if self.rep_cue_count >= self.max_cues_per_rep:
                return {"selected_error": None, "feedback": None, "deferred_errors": ordered, "decision_reason": "Rep cue limit reached.", "live_feedback": describe_live_feedback(None, phase, reason="Rep cue limit reached.")}

            self.last_error_time[error] = timestamp
            self.phase_counts[phase] = phase_count + 1
            self.rep_cue_count += 1
            return {
                "selected_error": error,
                "feedback": self.FEEDBACK.get(error),
                "deferred_errors": deferred,
                "decision_reason": f"Selected {error} by {phase} priority.",
                "live_feedback": describe_live_feedback(error, phase, self.FEEDBACK.get(error), reason=f"Selected {error} by {phase} priority."),
            }

        return {"selected_error": None, "feedback": None, "deferred_errors": ordered, "decision_reason": "Cooldown suppresses repeated cues.", "live_feedback": describe_live_feedback(None, phase, reason="Cooldown suppresses repeated cues.")}
