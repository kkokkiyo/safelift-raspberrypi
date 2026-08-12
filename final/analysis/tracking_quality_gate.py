from __future__ import annotations

from statistics import mean

from .frame_state import FrameState


LANDMARK_NAMES = {
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

# Nose/shoulders prove that the upper body is present; hips through foot points
# prove that the camera contains the complete lower body needed for a squat.
# The nose is useful for display, but it is not required to prove that the
# lower body is present. MediaPipe commonly drops face visibility while the
# legs remain perfectly usable for squat analysis.
FULL_BODY_INDICES = (11, 12, *tuple(LANDMARK_NAMES.keys()))


class TrackingQualityGate:
    """Quality gate that decides whether squat posture analysis is trustworthy."""

    def __init__(self, visibility_threshold: float = 0.6, presence_threshold: float = 0.6) -> None:
        self.visibility_threshold = visibility_threshold
        self.presence_threshold = presence_threshold

    def pose_is_usable(self, landmarks: dict[int, dict] | None, confidence: float) -> bool:
        """Reject face/torso-only frames before phase or rep processing."""
        if not landmarks or confidence < self.visibility_threshold:
            return False
        points = []
        for index in FULL_BODY_INDICES:
            landmark = landmarks.get(index)
            if not landmark:
                return False
            visibility = float(landmark.get("visibility", 0.0))
            x = float(landmark.get("x", -1.0))
            y = float(landmark.get("y", -1.0))
            if visibility < self.visibility_threshold or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return False
            points.append(y)

        hip_y = mean(float(landmarks[index]["y"]) for index in (23, 24))
        foot_y = mean(float(landmarks[index]["y"]) for index in (27, 28, 29, 30, 31, 32))
        return foot_y - hip_y >= 0.12 and max(points) - min(points) >= 0.22

    def pose_is_sendable(self, landmarks: dict[int, dict] | None, confidence: float) -> bool:
        """Gate pose packets for avatar retargeting separately from squat analysis.

        GIST_AVATA deliberately used a more tolerant streaming gate than its
        standing/calibration gate.  A single ankle visibility dip should not
        stop the avatar stream; the analytic IK and Unity hold-last-pose logic
        handle short gaps.  This must still reject torso-only/background poses.
        """
        # This is the avatar transport gate, not the scoring/calibration gate.
        # The old version duplicated the strict standing geometry checks below
        # and rejected every live frame in the final scene even though Unity
        # had all 33 landmarks. Keep only the minimum anatomical data required
        # by the Unity driver; the driver performs its own plausibility check
        # and holds the last safe pose when a packet is unusable.
        if not landmarks or confidence < 0.05:
            return False
        # Avatar transport only needs a plausible torso anchor. Lower-body
        # trust remains separate and is still decided by pose_is_usable().
        # This prevents one temporarily weak ankle from stopping the entire
        # avatar stream and lets Unity keep the upper body responsive while it
        # waits for a valid standing calibration.
        for index in (23, 24):
            point = landmarks.get(index)
            if not point or float(point.get("visibility", 0.0)) < 0.03:
                return False
            if not (-0.05 <= float(point.get("x", -1.0)) <= 1.05 and
                    -0.05 <= float(point.get("y", -1.0)) <= 1.12):
                return False
        return True

    def evaluate(self, frame_states: list[FrameState], rep_summary: dict | None = None) -> dict:
        if not frame_states:
            return self._result(
                ok=False,
                quality_level="bad",
                reason="분석할 프레임이 아직 충분하지 않습니다.",
                missing_landmarks=list(LANDMARK_NAMES.values()),
            )

        confidences = [state.confidence for state in frame_states]
        avg_confidence = mean(confidences) if confidences else 0.0

        landmark_visibilities: dict[str, list[float]] = {name: [] for name in LANDMARK_NAMES.values()}
        for state in frame_states:
            for idx, name in LANDMARK_NAMES.items():
                landmark = state.landmarks.get(idx)
                visibility = float(landmark.get("visibility", 0.0)) if landmark else 0.0
                landmark_visibilities[name].append(visibility)

        missing_landmarks = []
        lower_visibility_values = []
        for name, values in landmark_visibilities.items():
            if not values:
                missing_landmarks.append(name)
                continue
            lower_visibility_values.extend(values)
            visible_ratio = sum(1 for value in values if value >= self.visibility_threshold) / len(values)
            if visible_ratio < self.presence_threshold:
                missing_landmarks.append(name)

        lower_visibility_mean = mean(lower_visibility_values) if lower_visibility_values else 0.0

        if not self.pose_is_usable(frame_states[-1].landmarks, frame_states[-1].confidence):
            return self._result(
                ok=False,
                quality_level="bad",
                reason="전신이 화면에 충분히 보이지 않습니다. 머리부터 양발까지 화면 안에 들어오도록 카메라를 조정해주세요.",
                missing_landmarks=missing_landmarks,
                avg_confidence=avg_confidence,
                lower_visibility_mean=lower_visibility_mean,
            )

        if lower_visibility_mean < self.visibility_threshold:
            return self._result(
                ok=False,
                quality_level="bad",
                reason=f"현재 화면에서 하체 관절 표시율이 낮습니다. 평균 표시율={lower_visibility_mean:.2f}",
                missing_landmarks=missing_landmarks,
                avg_confidence=avg_confidence,
                lower_visibility_mean=lower_visibility_mean,
            )
        if missing_landmarks:
            return self._result(
                ok=False,
                quality_level="bad",
                reason="현재 화면에서 일부 하체 관절이 충분히 보이지 않습니다.",
                missing_landmarks=missing_landmarks,
                avg_confidence=avg_confidence,
                lower_visibility_mean=lower_visibility_mean,
            )
        if avg_confidence < 0.6:
            return self._result(
                ok=False,
                quality_level="bad",
                reason=f"현재 화면의 전체 자세 추적 신뢰도가 낮습니다. 평균 신뢰도={avg_confidence:.2f}",
                missing_landmarks=missing_landmarks,
                avg_confidence=avg_confidence,
                lower_visibility_mean=lower_visibility_mean,
            )
        if avg_confidence < 0.75:
            return self._result(
                ok=True,
                quality_level="weak",
                reason=f"분석은 가능하지만 현재 자세 추적 신뢰도가 다소 낮습니다. 평균 신뢰도={avg_confidence:.2f}",
                missing_landmarks=missing_landmarks,
                avg_confidence=avg_confidence,
                lower_visibility_mean=lower_visibility_mean,
            )
        return self._result(
            ok=True,
            quality_level="good",
            reason="현재 화면에서 하체 관절과 전체 자세가 안정적으로 추적되고 있습니다.",
            missing_landmarks=missing_landmarks,
            avg_confidence=avg_confidence,
            lower_visibility_mean=lower_visibility_mean,
        )

    def _result(
        self,
        ok: bool,
        quality_level: str,
        reason: str,
        missing_landmarks: list[str],
        avg_confidence: float = 0.0,
        lower_visibility_mean: float = 0.0,
    ) -> dict:
        return {
            "ok": ok,
            "quality_level": quality_level,
            "reason": reason,
            "missing_landmarks": missing_landmarks,
            "recommendation": (
                "전신이 화면에 들어오도록 카메라를 더 멀리 두고, 골반/무릎/발목/발끝이 보이는 구도로 다시 측정해주세요."
                if not ok
                else "현재 구도에서 분석을 진행할 수 있습니다."
            ),
            "avg_confidence": round(avg_confidence, 3),
            "lower_visibility_mean": round(lower_visibility_mean, 3),
        }
