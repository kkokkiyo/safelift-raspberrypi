import math
from typing import Any

import numpy as np


def safe_get_landmark(landmarks: dict[int, dict[str, float]], idx: int) -> dict[str, float] | None:
    return landmarks.get(idx)


def _point(landmark: dict[str, float] | None) -> np.ndarray | None:
    if landmark is None:
        return None
    return np.array([float(landmark["x"]), float(landmark["y"])], dtype=float)


def calculate_angle(a: Any, b: Any, c: Any) -> float | None:
    pa = _point(a) if isinstance(a, dict) else np.array(a, dtype=float) if a is not None else None
    pb = _point(b) if isinstance(b, dict) else np.array(b, dtype=float) if b is not None else None
    pc = _point(c) if isinstance(c, dict) else np.array(c, dtype=float) if c is not None else None
    if pa is None or pb is None or pc is None:
        return None
    ba = pa - pb
    bc = pc - pb
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return None
    cosine = float(np.dot(ba, bc) / norm)
    cosine = max(-1.0, min(1.0, cosine))
    return float(math.degrees(math.acos(cosine)))


def midpoint(p1: dict[str, float] | np.ndarray | None, p2: dict[str, float] | np.ndarray | None) -> tuple[float, float] | None:
    pp1 = _point(p1) if isinstance(p1, dict) else p1
    pp2 = _point(p2) if isinstance(p2, dict) else p2
    if pp1 is None or pp2 is None:
        return None
    mid = (np.array(pp1, dtype=float) + np.array(pp2, dtype=float)) / 2.0
    return float(mid[0]), float(mid[1])


def distance(p1: dict[str, float] | None, p2: dict[str, float] | None) -> float | None:
    pp1 = _point(p1)
    pp2 = _point(p2)
    if pp1 is None or pp2 is None:
        return None
    return float(np.linalg.norm(pp1 - pp2))


def compute_torso_angle(landmarks: dict[int, dict[str, float]]) -> float | None:
    shoulder_mid = midpoint(safe_get_landmark(landmarks, 11), safe_get_landmark(landmarks, 12))
    hip_mid = midpoint(safe_get_landmark(landmarks, 23), safe_get_landmark(landmarks, 24))
    if shoulder_mid is None or hip_mid is None:
        return None
    vector = np.array(shoulder_mid) - np.array(hip_mid)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    vertical_up = np.array([0.0, -1.0])
    cosine = float(np.dot(vector, vertical_up) / norm)
    cosine = max(-1.0, min(1.0, cosine))
    return float(math.degrees(math.acos(cosine)))


def compute_knee_angles(landmarks: dict[int, dict[str, float]]) -> dict[str, float | None]:
    left = calculate_angle(safe_get_landmark(landmarks, 23), safe_get_landmark(landmarks, 25), safe_get_landmark(landmarks, 27))
    right = calculate_angle(safe_get_landmark(landmarks, 24), safe_get_landmark(landmarks, 26), safe_get_landmark(landmarks, 28))
    values = [v for v in (left, right) if v is not None]
    return {
        "left_knee_angle": left,
        "right_knee_angle": right,
        "avg_knee_angle": sum(values) / len(values) if values else None,
    }


def compute_hip_height(landmarks: dict[int, dict[str, float]]) -> float | None:
    hip_mid = midpoint(safe_get_landmark(landmarks, 23), safe_get_landmark(landmarks, 24))
    return hip_mid[1] if hip_mid else None


def compute_knee_valgus_score(landmarks: dict[int, dict[str, float]]) -> float | None:
    left_knee = safe_get_landmark(landmarks, 25)
    right_knee = safe_get_landmark(landmarks, 26)
    left_ankle = safe_get_landmark(landmarks, 27)
    right_ankle = safe_get_landmark(landmarks, 28)
    left_foot = safe_get_landmark(landmarks, 31)
    right_foot = safe_get_landmark(landmarks, 32)
    hip_mid = midpoint(safe_get_landmark(landmarks, 23), safe_get_landmark(landmarks, 24))
    hip_width = distance(safe_get_landmark(landmarks, 23), safe_get_landmark(landmarks, 24))
    if not all([left_knee, right_knee, left_ankle, right_ankle, left_foot, right_foot, hip_mid, hip_width]):
        return None
    left_ref_x = (left_ankle["x"] + left_foot["x"]) / 2.0
    right_ref_x = (right_ankle["x"] + right_foot["x"]) / 2.0
    center_x = hip_mid[0]

    def toward_center(knee_x: float, foot_x: float) -> float:
        return knee_x - foot_x if foot_x < center_x else foot_x - knee_x

    score = max(toward_center(left_knee["x"], left_ref_x), toward_center(right_knee["x"], right_ref_x))
    return max(0.0, float(score / max(hip_width, 1e-6)))


def compute_pose_metrics(landmarks: dict[int, dict[str, float]], previous_hip_height: float | None = None, timestamp_delta: float | None = None) -> dict:
    knee = compute_knee_angles(landmarks)
    hip_height = compute_hip_height(landmarks)
    hip_velocity = None
    if hip_height is not None and previous_hip_height is not None and timestamp_delta and timestamp_delta > 0:
        hip_velocity = (hip_height - previous_hip_height) / timestamp_delta
    return {
        **knee,
        "torso_angle": compute_torso_angle(landmarks),
        "hip_height": hip_height,
        "hip_velocity": hip_velocity,
        "knee_valgus_score": compute_knee_valgus_score(landmarks),
    }
