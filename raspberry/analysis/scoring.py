from __future__ import annotations

from typing import Any


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def score_core(base: float | None, current: float | None) -> float:
    if base is None or current is None or base <= 0:
        return 35.0
    drop = max(0.0, (base - current) / base * 100.0)
    return round(clamp(35.0 if drop <= 10 else 35 - (drop - 10) * 1.16, 0, 35), 1)


def score_balance(left: float | None, right: float | None) -> tuple[float, float | None]:
    if left is None or right is None:
        return 35.0, None
    total = max(left + right, 1e-6)
    imbalance = abs(left - right) / total * 100.0
    imbalance_score = clamp(23 if imbalance <= 10 else 23 - (imbalance - 10) * 0.6, 0, 23)
    # Type-A rubric keeps 12 points for heel/support maintenance. The direct
    # FSR reader has no separate heel-drop flag yet, so retain those points
    # provisionally and expose the missing evidence in the future extension.
    return round(imbalance_score + 12.0, 1), round(imbalance, 2)


def score_spine(mp_angle: float | None, imu_angle: float | None, offset: float = 0.0) -> tuple[float, float | None]:
    if mp_angle is None or imu_angle is None:
        return 30.0, None
    delta = abs((mp_angle - imu_angle) - offset)
    return round(clamp(30 if delta <= 5 else 30 - (delta - 5) * 2, 0, 30), 1), round(delta, 2)


def evaluate(frame: dict[str, Any], pose: dict[str, Any] | None, baseline: dict) -> dict:
    balance, imbalance = score_balance(frame.get("left_pressure_ratio"), frame.get("right_pressure_ratio"))
    core = score_core(baseline.get("core_pressure_ratio"), frame.get("core_pressure_ratio"))
    mp_angle = (pose or {}).get("torso_angle")
    spine, delta = score_spine(mp_angle, frame.get("imu_pitch"), (baseline.get("imu_pitch") or 0) - ((pose or {}).get("baseline_torso_angle") or 0))
    total = round(core + balance + spine, 1)
    if total >= 90: grade, label = "S", "우수"
    elif total >= 80: grade, label = "A", "안정"
    elif total >= 65: grade, label = "B", "자세 교정 필요"
    else: grade, label = "C", "위험 경고"
    return {"score": int(round(total)), "score_raw": total, "grade": grade, "label": label, "core": core, "balance": balance, "spine": spine, "imbalance_percent": imbalance, "spine_delta_degrees": delta, "evidence": "complete" if mp_angle is not None and frame.get("imu_pitch") is not None else "partial"}
