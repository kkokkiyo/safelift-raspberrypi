"""SafeLift XR 100-point scoring rubric.

Rubric from the user's scoring table:
- Core bracing / intra-abdominal pressure: 35 pts
- Ground balance / FSR: 35 pts
- Spine alignment / butt wink: 30 pts

The module works with the normalized backend contract. Raw sensor acquisition is
outside the implementation scope; provider code must normalize values before
they reach this rubric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _stat(summary: dict, key: str, stat: str, default: float | None = None) -> float | None:
    value = (summary or {}).get(key)
    if isinstance(value, dict):
        return _num(value.get(stat), default)
    return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round_score(value: float, max_points: float) -> float:
    return round(_clamp(value, 0.0, max_points), 1)


def core_bracing_score(core_base: float | None, core_min: float | None) -> dict:
    """35-point air-pressure belt score.

    Drop Ratio (%) = (P_base - P_min) / P_base * 100
    <=10% drop: full 35 pts.
    >10% drop: subtract 1.16 pts per excess 1%.
    """

    base = _num(core_base)
    min_value = _num(core_min)
    if base is None or base <= 0 or min_value is None:
        missing = []
        if base is None or base <= 0:
            missing.append("core_pressure_baseline")
        if min_value is None:
            missing.append("core_pressure_min")
        return {
            "score": 35.0,
            "max_points": 35,
            "drop_ratio_percent": None,
            "evidence_complete": False,
            "missing_evidence": missing,
            "formula": "35 - max(0, drop_ratio_percent - 10) * 1.16; score is provisional because normalized core evidence is incomplete",
        }

    drop_ratio = max(0.0, (base - min_value) / max(base, 1e-6) * 100.0)
    score = 35.0 if drop_ratio <= 10.0 else 35.0 - (drop_ratio - 10.0) * 1.16
    return {
        "score": _round_score(score, 35.0),
        "max_points": 35,
        "drop_ratio_percent": round(drop_ratio, 2),
        "evidence_complete": True,
        "missing_evidence": [],
        "formula": "35 - max(0, drop_ratio_percent - 10) * 1.16",
    }


def fsr_balance_score(
    left_total: float | None,
    right_total: float | None,
    heel_maintained: bool | None = None,
    type_b_forefoot_shift_percent: float | None = None,
    scoring_type: str = "A",
) -> dict:
    """35-point FSR balance score.

    Type A:
      - left/right imbalance: 23 pts, <=10% full, -0.6 per excess 1%
      - heel maintain: 12 pts, zero if heel pressure drops below threshold
    Type B:
      - corrected left/right imbalance: same 23 pts
      - forefoot stability: 12 pts, <=100% full, -1.2 per excess 10%
    """

    left = _num(left_total)
    right = _num(right_total)
    missing: list[str] = []
    if left is None:
        missing.append("left_pressure_ratio")
    if right is None:
        missing.append("right_pressure_ratio")
    if left is None or right is None:
        imbalance_percent = None
        imbalance_score = 23.0
    else:
        total = max(left + right, 1e-6)
        imbalance_percent = abs(left - right) / total * 100.0
        imbalance_score = 23.0 if imbalance_percent <= 10.0 else 23.0 - (imbalance_percent - 10.0) * 0.6

    mode = (scoring_type or "A").strip().upper()

    if mode == "B":
        shift = _num(type_b_forefoot_shift_percent)
        if shift is None:
            missing.append("forefoot_shift_percent")
            shift = 0.0
        forefoot_score = 12.0 if shift <= 100.0 else 12.0 - ((shift - 100.0) / 10.0) * 1.2
        support_score = forefoot_score
        support_name = "forefoot_stability_score"
    else:
        if heel_maintained is None:
            missing.append("heel_pressure_drop_or_heel_maintained")
        support_score = 0.0 if heel_maintained is False else 12.0
        support_name = "heel_maintain_score"

    score = _round_score(imbalance_score, 23.0) + _round_score(support_score, 12.0)
    return {
        "score": _round_score(score, 35.0),
        "max_points": 35,
        "scoring_type": mode,
        "imbalance_percent": None if imbalance_percent is None else round(imbalance_percent, 2),
        "imbalance_score": _round_score(imbalance_score, 23.0),
        support_name: _round_score(support_score, 12.0),
        "evidence_complete": not missing,
        "missing_evidence": missing,
        "formula": "imbalance 23pts: <=10% full, -0.6/excess%; support 12pts by Type A heel or Type B forefoot stability",
    }


def spine_alignment_score(media_pipe_angle: float | None, imu_angle: float | None, base_offset: float | None = 0.0) -> dict:
    """30-point spine/pelvis alignment score.

    Delta theta max = abs((MediaPipe angle - IMU angle) - O_base)
    <=5 degrees: full 30 pts.
    >5 degrees: subtract 2 pts per excess degree.
    """

    mp = _num(media_pipe_angle)
    imu = _num(imu_angle)
    offset = _num(base_offset, 0.0) or 0.0
    if mp is None or imu is None:
        missing = []
        if mp is None:
            missing.append("mediapipe_torso_angle")
        if imu is None:
            missing.append("imu_pitch")
        return {
            "score": 30.0,
            "max_points": 30,
            "delta_theta_degrees": None,
            "evidence_complete": False,
            "missing_evidence": missing,
            "formula": "30 - max(0, delta_theta_degrees - 5) * 2; score is provisional because synchronized angle evidence is incomplete",
        }

    delta = abs((mp - imu) - offset)
    score = 30.0 if delta <= 5.0 else 30.0 - (delta - 5.0) * 2.0
    return {
        "score": _round_score(score, 30.0),
        "max_points": 30,
        "delta_theta_degrees": round(delta, 2),
        "evidence_complete": True,
        "missing_evidence": [],
        "formula": "30 - max(0, delta_theta_degrees - 5) * 2",
    }


def grade_for(score: float) -> dict:
    score = float(score)
    if score >= 90:
        return {"grade": "S", "label": "파워리프트 마스터", "message": "복압, 좌우 밸런스, 척추 중립을 안정적으로 유지한 정석 스쿼트입니다."}
    if score >= 80:
        return {"grade": "A", "label": "안정적인 스쿼트", "message": "좋은 자세입니다. 하강 중 미세한 복압 유지 또는 좌우 체중 분산만 보완해보세요."}
    if score >= 65:
        return {"grade": "B", "label": "자세 교정 필요", "message": "최하단 구간에서 브레이싱, 발 지지, 척추 정렬을 다시 확인하세요."}
    return {"grade": "C", "label": "부상 위험 경고", "message": "복압 유지와 좌우 50:50 하중을 다시 연습한 뒤 진행하세요."}

def build_rubric_score_summary(rep_summary: dict, set_comparison: dict | None = None) -> dict:
    metrics = rep_summary.get("metrics") or {}
    sensor_summary = rep_summary.get("sensor_summary") or {}
    calibration = rep_summary.get("calibration_baseline") or {}
    set2 = (set_comparison or {}).get("set2") or {}
    set2_metrics = set2.get("metrics") or {}
    set2_sensor = set2.get("sensor_summary") or {}

    core_base = _stat(calibration, "core_pressure_ratio", "mean")
    if core_base is None:
        core_base = _stat(sensor_summary, "core_pressure_ratio", "max")
    core_min = metrics.get("core_pressure_min_ratio")
    if core_min is None:
        core_min = _stat(sensor_summary, "core_pressure_ratio", "min")
    if core_min is None:
        core_min = _stat(set2_sensor, "core_pressure_ratio", "min")

    left = _stat(sensor_summary, "left_pressure_ratio", "mean")
    right = _stat(sensor_summary, "right_pressure_ratio", "mean")
    if left is None:
        left = _stat(set2_sensor, "left_pressure_ratio", "mean")
    if right is None:
        right = _stat(set2_sensor, "right_pressure_ratio", "mean")
    heel_drop = bool(metrics.get("heel_pressure_drop") or sensor_summary.get("heel_pressure_drop_count") or set2_sensor.get("heel_pressure_drop_count"))
    fsr_scoring_type = str(
        metrics.get("fsr_scoring_type")
        or sensor_summary.get("fsr_scoring_type")
        or set2_metrics.get("fsr_scoring_type")
        or set2_sensor.get("fsr_scoring_type")
        or "A"
    ).upper()
    if fsr_scoring_type not in {"A", "B"}:
        fsr_scoring_type = "A"
    forefoot_shift_percent = (
        metrics.get("forefoot_shift_peak_percent")
        or _stat(sensor_summary, "forefoot_shift_percent", "max")
        or set2_metrics.get("forefoot_shift_peak_percent")
        or _stat(set2_sensor, "forefoot_shift_percent", "max")
    )

    # Use the sagittal side-camera angle for spine alignment when available.
    # The front camera remains the authoritative squat phase/rep source.
    mp_angle = metrics.get("max_side_torso_angle") or metrics.get("max_torso_angle")
    if mp_angle is None:
        mp_angle = set2_metrics.get("torso_angle_peak_abs") or set2_metrics.get("torso_angle_mean")
    imu_angle = metrics.get("imu_pitch_peak")
    if imu_angle is None:
        imu_angle = _stat(sensor_summary, "imu_pitch", "max")
    if imu_angle is None:
        imu_angle = _stat(set2_sensor, "imu_pitch", "max")

    core = core_bracing_score(core_base, core_min)
    balance = fsr_balance_score(
        left,
        right,
        heel_maintained=not heel_drop,
        type_b_forefoot_shift_percent=forefoot_shift_percent,
        scoring_type=fsr_scoring_type,
    )
    spine = spine_alignment_score(mp_angle, imu_angle, 0.0)
    components = [core, balance, spine]
    missing_score_evidence: list[str] = []
    for component in components:
        missing_score_evidence.extend(component.get("missing_evidence") or [])
    score_evidence_complete = all(component.get("evidence_complete", True) for component in components)
    score_confidence = "complete" if score_evidence_complete else "partial"
    total = round(core["score"] + balance["score"] + spine["score"], 1)
    grade = grade_for(total)

    return {
        "safe_lift_score": int(round(total)),
        "safe_lift_score_raw": total,
        "core_bracing_score": int(round(core["score"])),
        "balance_score": int(round(balance["score"])),
        "spine_alignment_score": int(round(spine["score"])),
        "torso_stability_score": int(round(spine["score"])),
        "knee_alignment_score": int(round(balance["score"])),
        "improvement_delta": int(max(0, -(set_comparison or {}).get("error_count_delta", 0)) * 3),
        "basis": "safelift_rubric_score_v1",
        "max_points": 100,
        "score_evidence_complete": score_evidence_complete,
        "score_confidence": score_confidence,
        "missing_score_evidence": sorted(set(missing_score_evidence)),
        "rubric": {
            "core_bracing": core,
            "ground_balance_fsr": balance,
            "spine_alignment": spine,
        },
        "grade": grade["grade"],
        "grade_label": grade["label"],
        "grade_message": grade["message"],
        "summary": (
            f"SafeLift Score {int(round(total))}/100 ({grade['grade']} · {grade['label']}). "
            "Basis: 35 core + 35 FSR balance + 30 spine alignment. "
            f"Evidence confidence: {score_confidence}."
        ),
    }
