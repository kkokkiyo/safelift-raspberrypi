from __future__ import annotations

import math


def angle(a, b, c) -> float:
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    den = max(1e-6, math.hypot(*ab) * math.hypot(*cb))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / den))))


class PoseAnalyzer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.pose = None
        if enabled:
            import mediapipe as mp
            self.mp = mp
            self.pose = mp.solutions.pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def analyze(self, frame):
        if not self.enabled or self.pose is None:
            return None, frame
        import cv2
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        if not result.pose_landmarks:
            return {"confidence": 0.0}, frame
        lm = result.pose_landmarks.landmark
        P = self.mp.solutions.pose.PoseLandmark
        def xy(index): return (lm[index].x, lm[index].y)
        hip = xy(P.LEFT_HIP.value); shoulder = xy(P.LEFT_SHOULDER.value)
        knee = xy(P.LEFT_KNEE.value); ankle = xy(P.LEFT_ANKLE.value)
        torso_angle = math.degrees(math.atan2(shoulder[0] - hip[0], hip[1] - shoulder[1]))
        metrics = {"confidence": round(sum(p.visibility for p in lm) / len(lm), 3), "knee_angle": round(angle(hip, knee, ankle), 2), "torso_angle": round(torso_angle, 2)}
        self.mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, self.mp.solutions.pose.POSE_CONNECTIONS)
        return metrics, frame

    def close(self):
        if self.pose:
            self.pose.close()
