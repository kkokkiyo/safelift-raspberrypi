from collections import deque

import numpy as np


class PhaseDetector:
    def __init__(
        self,
        history_size: int = 18,
        standing_knee_angle: float = 145.0,
        bottom_knee_angle: float = 130.0,
        min_motion_delta: float = 0.004,
        bottom_near_max: float = 0.024,
    ) -> None:
        self.history = deque(maxlen=history_size)
        self.phase_history = deque(maxlen=8)
        self.standing_knee_angle = standing_knee_angle
        self.bottom_knee_angle = bottom_knee_angle
        self.min_motion_delta = min_motion_delta
        self.bottom_near_max = bottom_near_max

    def reset(self) -> None:
        self.history.clear()
        self.phase_history.clear()

    def update(self, metrics: dict) -> str:
        hip_height = metrics.get("hip_height")
        avg_knee_angle = metrics.get("avg_knee_angle")
        if hip_height is None or avg_knee_angle is None:
            phase = self.phase_history[-1] if self.phase_history else "Unknown"
            self.phase_history.append(phase)
            return phase

        self.history.append(float(hip_height))
        if len(self.history) < 4:
            phase = "Standing" if avg_knee_angle > self.standing_knee_angle else "Unknown"
            self.phase_history.append(phase)
            return phase

        history = list(self.history)
        previous = history[:-1][-5:]
        recent_average = float(np.mean(previous)) if previous else float(hip_height)
        motion_delta = float(hip_height) - recent_average
        max_hip_y = max(history)
        near_bottom = float(hip_height) >= max_hip_y - self.bottom_near_max

        moving_down = motion_delta > self.min_motion_delta
        moving_up = motion_delta < -self.min_motion_delta

        if avg_knee_angle > self.standing_knee_angle and abs(motion_delta) < self.min_motion_delta:
            phase = "Standing"
        elif moving_down and avg_knee_angle > self.bottom_knee_angle:
            phase = "Descent"
        elif avg_knee_angle < self.bottom_knee_angle or (near_bottom and not moving_down and avg_knee_angle <= self.standing_knee_angle):
            phase = "Bottom"
        elif moving_up:
            phase = "Ascent"
        else:
            phase = self.phase_history[-1] if self.phase_history else "Standing"

        self.phase_history.append(phase)
        return phase
