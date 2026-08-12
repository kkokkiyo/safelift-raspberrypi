import cv2
import mediapipe as mp
import numpy as np


KEY_LANDMARKS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


class MediaPipePoseTracker:
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5) -> None:
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame) -> tuple[dict, float, np.ndarray]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()
        if results.pose_landmarks is None:
            return {}, 0.0, annotated

        self.mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        landmarks = {
            idx: {
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(lm.visibility),
            }
            for idx, lm in enumerate(results.pose_landmarks.landmark)
        }
        confidence = float(np.mean([landmarks[idx]["visibility"] for idx in KEY_LANDMARKS if idx in landmarks]))
        return landmarks, confidence, annotated

    def close(self) -> None:
        self.pose.close()
