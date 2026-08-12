from dataclasses import dataclass


@dataclass
class RepCounterSnapshot:
    total_reps: int
    set_index: int
    set_rep_count: int
    phase: str
    last_completed: bool
    tracking_accepted: bool
    blocked_by_tracking: bool


class SquatRepCounter:
    """Counts only physically plausible squat repetitions from MediaPipe.

    The previous counter accepted any Descent/Bottom/Ascent -> Standing phase
    transition. That was too permissive: camera jitter or a mock phase stream
    could fill SET 1/SET 2 without a real squat. This counter requires:

    - ordered phase progression: Standing -> Descent -> Bottom -> Ascent -> Standing
    - accepted lower-body tracking while the rep is being built
    - a minimum movement amplitude from standing to bottom
    - a minimum rep duration so one-frame phase spikes cannot count
    """

    BLOCKED_TRACKING_TOKENS = {"bad", "uncertain", "lost", "poor", "blocked", "unknown", "none"}

    def __init__(
        self,
        reps_per_set: int = 3,
        # Moderate thresholds: still require both hip descent and knee flexion,
        # but accept a natural shallow squat instead of demanding a deep,
        # carefully staged movement from the user.
        min_hip_drop: float = 0.018,
        min_knee_drop_degrees: float = 7.0,
        min_rep_seconds: float = 0.30,
        min_descent_frames: int = 1,
        min_bottom_frames: int = 1,
        min_ascent_frames: int = 1,
        max_tracking_gap_frames: int = 12,
    ) -> None:
        self.reps_per_set = max(1, reps_per_set)
        self.min_hip_drop = max(0.0, min_hip_drop)
        self.min_knee_drop_degrees = max(0.0, min_knee_drop_degrees)
        self.min_rep_seconds = max(0.0, min_rep_seconds)
        self.min_descent_frames = max(1, min_descent_frames)
        self.min_bottom_frames = max(1, min_bottom_frames)
        self.min_ascent_frames = max(1, min_ascent_frames)
        self.max_tracking_gap_frames = max(0, max_tracking_gap_frames)
        self.reset()

    def reset(self) -> None:
        self.total_reps = 0
        self._last_phase = "Unknown"
        self._reset_candidate()

    def _reset_candidate(self) -> None:
        self._stage = "idle"
        self._standing_hip_height = None
        self._standing_knee_angle = None
        self._max_hip_height = None
        self._min_knee_angle = None
        self._candidate_started_at = None
        self._descent_frames = 0
        self._bottom_frames = 0
        self._ascent_frames = 0
        self._tracking_gap_frames = 0

    def _make_snapshot(self, completed: bool, tracking_accepted: bool, blocked_by_tracking: bool) -> RepCounterSnapshot:
        set_index = min(2, (self.total_reps // self.reps_per_set) + 1)
        set_rep_count = self.total_reps % self.reps_per_set
        if set_rep_count == 0 and self.total_reps > 0:
            set_rep_count = self.reps_per_set
        return RepCounterSnapshot(
            total_reps=self.total_reps,
            set_index=set_index,
            set_rep_count=set_rep_count,
            phase=self._last_phase,
            last_completed=completed,
            tracking_accepted=tracking_accepted,
            blocked_by_tracking=blocked_by_tracking,
        )

    def _tracking_is_accepted(self, tracking_quality: object) -> bool:
        if isinstance(tracking_quality, dict):
            # TrackingQualityGate returns a structured result.  The previous
            # implementation ignored `ok` and `quality_level`, so a dict such
            # as {ok: false, quality_level: "bad"} could be normalized only
            # from its localized reason and accidentally accepted.
            if tracking_quality.get("ok") is False:
                return False
            quality_level = tracking_quality.get("quality_level")
            if quality_level is not None:
                normalized_level = str(quality_level).strip().lower()
                if normalized_level not in {"good", "weak"}:
                    return False
            parts = [
                quality_level,
                tracking_quality.get("quality"),
                tracking_quality.get("status"),
                tracking_quality.get("level"),
                tracking_quality.get("tracking_quality"),
                tracking_quality.get("reason"),
            ]
            normalized = " ".join(str(part) for part in parts if part is not None).strip().lower()
            if not normalized:
                normalized = "good"
        else:
            normalized = str(tracking_quality or "GOOD").strip().lower()
        return not any(token in normalized for token in self.BLOCKED_TRACKING_TOKENS)

    def _metric_float(self, metrics: dict | None, key: str) -> float | None:
        if not metrics:
            return None
        value = metrics.get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _update_extremes(self, metrics: dict | None) -> None:
        hip_height = self._metric_float(metrics, "hip_height")
        knee_angle = self._metric_float(metrics, "avg_knee_angle")
        if hip_height is not None:
            if self._max_hip_height is None or hip_height > self._max_hip_height:
                self._max_hip_height = hip_height
        if knee_angle is not None:
            if self._min_knee_angle is None or knee_angle < self._min_knee_angle:
                self._min_knee_angle = knee_angle

    def _candidate_is_valid(self, timestamp: float | None) -> bool:
        if self._descent_frames < self.min_descent_frames:
            return False
        if self._bottom_frames < self.min_bottom_frames:
            return False
        if self._ascent_frames < self.min_ascent_frames:
            return False
        if timestamp is not None and self._candidate_started_at is not None:
            if timestamp - self._candidate_started_at < self.min_rep_seconds:
                return False

        hip_drop = None
        if self._standing_hip_height is not None and self._max_hip_height is not None:
            # MediaPipe normalized y increases downward, so a squat lowers the hip by increasing y.
            hip_drop = self._max_hip_height - self._standing_hip_height

        knee_drop = None
        if self._standing_knee_angle is not None and self._min_knee_angle is not None:
            knee_drop = self._standing_knee_angle - self._min_knee_angle

        moved_enough_by_hip = hip_drop is not None and hip_drop >= self.min_hip_drop
        moved_enough_by_knee = knee_drop is not None and knee_drop >= self.min_knee_drop_degrees
        # A valid squat must show both meaningful hip descent and knee flexion.
        # Accepting either signal alone permits camera jitter or partial-body
        # motion to complete a repetition.
        return moved_enough_by_hip and moved_enough_by_knee

    def update(
        self,
        phase: str,
        tracking_quality: object = "GOOD",
        metrics: dict | None = None,
        timestamp: float | None = None,
    ) -> RepCounterSnapshot:
        self._last_phase = phase or "Unknown"
        tracking_accepted = self._tracking_is_accepted(tracking_quality)
        if not tracking_accepted:
            self._tracking_gap_frames += 1
            if self._tracking_gap_frames > self.max_tracking_gap_frames:
                self._reset_candidate()
            return self._make_snapshot(False, False, True)
        self._tracking_gap_frames = 0

        hip_height = self._metric_float(metrics, "hip_height")
        knee_angle = self._metric_float(metrics, "avg_knee_angle")
        normalized = self._last_phase.strip().lower()
        # MediaPipe can briefly report an unknown phase while the pose is
        # still usable. If the person is clearly upright after an ascent,
        # treat that frame as the required return to standing instead of
        # discarding an otherwise valid repetition.
        if normalized in {"unknown", "none", ""} and self._stage == "ascent":
            if knee_angle is not None and knee_angle >= 145.0:
                normalized = "standing"
        completed = False

        if normalized == "standing":
            if self._stage == "idle":
                if hip_height is not None:
                    self._standing_hip_height = hip_height
                if knee_angle is not None:
                    self._standing_knee_angle = knee_angle
            elif self._stage == "ascent":
                if self._candidate_is_valid(timestamp):
                    self.total_reps += 1
                    completed = True
                self._reset_candidate()
                if hip_height is not None:
                    self._standing_hip_height = hip_height
                if knee_angle is not None:
                    self._standing_knee_angle = knee_angle
            return self._make_snapshot(completed, True, False)

        if normalized == "descent":
            if self._stage == "idle":
                self._stage = "descent"
                self._candidate_started_at = timestamp
                if self._standing_hip_height is None and hip_height is not None:
                    self._standing_hip_height = hip_height
                if self._standing_knee_angle is None and knee_angle is not None:
                    self._standing_knee_angle = knee_angle
            if self._stage in {"descent", "bottom", "ascent"}:
                self._descent_frames += 1
                self._update_extremes(metrics)
            return self._make_snapshot(False, True, False)

        if normalized == "bottom":
            if self._stage in {"descent", "bottom"}:
                self._stage = "bottom"
                self._bottom_frames += 1
                self._update_extremes(metrics)
            else:
                # Bottom without a prior descent is a tracking spike, not a rep.
                self._reset_candidate()
            return self._make_snapshot(False, True, False)

        if normalized == "ascent":
            if self._stage in {"bottom", "ascent"}:
                self._stage = "ascent"
                self._ascent_frames += 1
                self._update_extremes(metrics)
            else:
                # Ascent without a confirmed bottom is not enough to count.
                self._reset_candidate()
            return self._make_snapshot(False, True, False)

        return self._make_snapshot(False, True, False)
