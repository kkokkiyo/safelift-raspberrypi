from __future__ import annotations


def assess(frame: dict, pose: dict | None, score: dict) -> dict:
    reasons = []
    if score.get("imbalance_percent") is not None and score["imbalance_percent"] > 20:
        reasons.append("좌우 발 압력 불균형")
    if frame.get("imu_pitch") is not None and abs(frame["imu_pitch"]) > 20:
        reasons.append("몸통 기울기 증가")
    if frame.get("imu_instability") is not None and frame["imu_instability"] > 0.65:
        reasons.append("IMU 흔들림 증가")
    if pose and pose.get("knee_angle") is not None and pose["knee_angle"] < 65:
        reasons.append("무릎 굽힘 과다")
    if not frame.get("valid", True):
        reasons.append("센서 입력 이상")
    level = "normal" if not reasons and score["score"] >= 80 else "warning" if score["score"] >= 60 else "danger"
    return {"level": level, "reasons": reasons, "message": "정상 자세입니다." if not reasons else ", ".join(reasons)}
