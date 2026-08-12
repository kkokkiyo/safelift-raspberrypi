import argparse
import json
import platform
import socket
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from analysis.frame_state import FrameState
from analysis.calibration_tracker import CalibrationTracker
from analysis.geometry import compute_pose_metrics
from analysis.phase_detector import PhaseDetector
from analysis.rep_quality import LastRepQualityEvaluator
from analysis.rep_summary_builder import RepSummaryBuilder
from analysis.risk_aware_feedback import RiskAwareFeedbackEngine
from analysis.live_feedback_contract import describe_live_feedback
from analysis.rule_error_detector import RuleErrorDetector
from analysis.score_rubric import build_rubric_score_summary
from analysis.tracking_quality_gate import TrackingQualityGate
from analysis.squat_rep_counter import SquatRepCounter
from camera.mediapipe_pose_tracker import MediaPipePoseTracker
from capture.phase_capture_manager import PhaseCaptureManager
from reports.gemini_reporter import GeminiReporter
from reports.quality_evidence import build_quality_evidence_text
from reports.template_reporter import TemplateReporter
from sensors.http_sensor_provider import HttpSensorProvider
from sensors.jsonl_sensor_provider import JsonlSensorProvider
from sensors.sensor_frame import SensorFrame
from sensors.serial_sensor_provider import SerialSensorProvider
from ui.dashboard import BrowserDashboard, DashboardRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SafeLift XR webcam + Gemini post-rep coaching backend")
    parser.add_argument("--camera", type=str, default="auto", help="OpenCV camera index or auto")
    parser.add_argument(
        "--camera-half",
        dest="camera_half",
        choices=["both", "left", "right"],
        default="both",
        help="Use the full camera frame or one half of an OBS side-by-side composite.",
    )
    parser.add_argument("--side_camera", type=str, default="", help="Optional side camera index; pose metrics only, no preview")
    parser.add_argument("--width", type=int, default=1280)
    # OBS composite is two 640x480 sources side by side.  Keep the
    # captured composite at 1280x480 so selecting one half produces the
    # native 640x480 camera image without vertical stretching.
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fourcc", type=str, default="MJPG")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--strict_tracking_quality",
        action="store_true",
        default=True,
        help="Use strict lower-body quality gates (default).",
    )
    parser.add_argument(
        "--relaxed_tracking_quality",
        action="store_true",
        help="Opt out of strict lower-body quality gates for development only.",
    )
    parser.add_argument("--rule_confidence_threshold", type=float, default=None, help="Override rule detector pose confidence threshold")
    parser.add_argument("--serial_port", type=str, default="")
    parser.add_argument("--sensor_jsonl", type=str, default="", help="Replay normalized sensor frames from a JSONL file; this is not raw hardware acquisition.")
    parser.add_argument("--sensor_jsonl_no_loop", action="store_true", help="Do not loop --sensor_jsonl when it reaches EOF.")
    parser.add_argument("--sensor_http", action="store_true", help="Accept normalized sensor frames through POST /api/sensor_frame on the browser backend.")
    parser.add_argument("--sensor_http_stale_seconds", type=float, default=2.0, help="Seconds before an HTTP-pushed sensor frame is considered stale.")
    parser.add_argument("--use_gemini", action="store_true", default=True,
                        help="Use Gemini API for coach reports (falls back to template).")
    parser.add_argument("--no_gemini", action="store_true", help="Disable Gemini and use template reports.")
    parser.add_argument("--gemini_model", type=str, default="gemini-flash-latest")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--keep_outputs", action="store_true", help="Keep captures/reports after the app exits")
    parser.add_argument("--browser", action="store_true", help="Show dashboard in browser instead of cv2.imshow only")
    parser.add_argument("--no_camera_window", action="store_true", help="Disable the local MediaPipe camera preview window")
    parser.add_argument("--avatar_udp_host", type=str, default="127.0.0.1", help="Unity MediaPipe avatar receiver host")
    parser.add_argument("--avatar_udp_port", type=int, default=5054, help="Unity MediaPipe avatar receiver UDP port")
    parser.add_argument("--no_avatar_stream", action="store_true", help="Disable MediaPipe landmark streaming to Unity avatar")
    parser.add_argument("--web_host", type=str, default="127.0.0.1")
    parser.add_argument("--web_port", type=int, default=8090)
    parser.add_argument("--report_share_root", type=str, default="", help="Laptop B shared folder for static token reports")
    parser.add_argument("--report_public_base", type=str, default="http://squartreport.ddnsfree.com/reports", help="Fixed public URL prefix for token reports")
    parser.add_argument("--no_flip", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.relaxed_tracking_quality:
        args.strict_tracking_quality = False
    return args


def trim_dark_padding(frame: np.ndarray) -> np.ndarray:
    """Remove black OBS letterbox padding before normalizing the camera frame."""
    if frame is None or frame.size == 0:
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # A row/column is considered content-bearing when enough pixels are
    # brighter than the near-black OBS background.
    active_rows = (np.mean(gray > 12, axis=1) > 0.03)
    active_cols = (np.mean(gray > 12, axis=0) > 0.03)
    row_indices = np.flatnonzero(active_rows)
    col_indices = np.flatnonzero(active_cols)
    if row_indices.size == 0 or col_indices.size == 0:
        return frame
    y0, y1 = int(row_indices[0]), int(row_indices[-1]) + 1
    x0, x1 = int(col_indices[0]), int(col_indices[-1]) + 1
    cropped = frame[y0:y1, x0:x1]
    # Reject implausibly small detections so a dark camera scene is not
    # accidentally cropped down to a tiny patch.
    if cropped.shape[0] < frame.shape[0] * 0.35 or cropped.shape[1] < frame.shape[1] * 0.35:
        return frame
    return cropped


def open_camera(camera: str, width: int, height: int, fourcc: str) -> tuple[cv2.VideoCapture, int]:
    candidates = list(range(10)) if camera.lower() == "auto" else [int(camera)]
    for index in candidates:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if fourcc and fourcc.upper() != "NONE":
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4].upper()))
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"Selected camera index: {index}")
                return cap, index
        cap.release()
    if "microsoft" in platform.release().lower():
        print("Camera open failed. If this is WSL, attach the webcam to WSL first.")
    else:
        print("Camera open failed. Check that no other app is using the webcam.")
    return cv2.VideoCapture(), -1


def create_sensor_provider(args: argparse.Namespace):
    if args.sensor_jsonl:
        try:
            print(f"Using JSONL sensor provider: {args.sensor_jsonl}")
            return JsonlSensorProvider(args.sensor_jsonl, loop=not args.sensor_jsonl_no_loop)
        except Exception as exc:
            raise RuntimeError(f"JSONL sensor provider unavailable: {exc}") from exc
    if args.sensor_http:
        print("Using HTTP sensor provider: POST normalized frames to /api/sensor_frame")
        return HttpSensorProvider(stale_after_seconds=args.sensor_http_stale_seconds)
    if args.serial_port:
        try:
            print(f"Using serial sensor provider: {args.serial_port}")
            return SerialSensorProvider(args.serial_port)
        except Exception as exc:
            raise RuntimeError(f"Serial sensor unavailable: {exc}") from exc
    raise RuntimeError(
        "No real sensor provider configured. Start the sensor bridge and use --serial_port COMx, "
        "or use --sensor_http with a real sensor POST producer. Mock sensors are disabled."
    )


def make_reporter(args: argparse.Namespace):
    template = TemplateReporter()
    if args.no_gemini:
        return template
    return GeminiReporter(fallback=template, model=args.gemini_model)


def frame_state_to_jsonable(state: FrameState) -> dict:
    return {
        "timestamp": state.timestamp,
        "frame_id": state.frame_id,
        "phase": state.phase,
        "confidence": state.confidence,
        "metrics": state.metrics,
        "sensor": asdict(state.sensor),
        "detected_errors": state.detected_errors,
        "realtime_feedback": state.realtime_feedback,
        "selected_error": state.selected_error,
        "deferred_errors": state.deferred_errors,
        "decision_reason": state.decision_reason,
        "live_feedback": describe_live_feedback(state.selected_error, state.phase, state.realtime_feedback, state.sensor, state.decision_reason),
        "rep_count": state.total_reps,
        "set_index": state.set_index,
        "set_rep_count": state.set_rep_count,
        "rep_completed": state.rep_completed,
        "calibration_status": state.calibration_status,
        "calibration_progress": state.calibration_progress,
        "calibration_message": state.calibration_message,
        "calibration_baseline": state.calibration_baseline,
    }


def load_phase_images(phase_snapshots: dict) -> dict:
    images = {}
    for phase, path in phase_snapshots.items():
        img = cv2.imread(str(path))
        images[phase] = img
    return images


def enrich_rep_summary_for_vlm(rep_summary: dict, frame_states: list[FrameState], last_rep_quality: dict) -> dict:
    rule = rep_summary.get("rule_analysis", {})
    selected_feedback = rule.get("selected_realtime_feedback") or []
    sensor_summary = build_sensor_summary(frame_states)
    sensor_mode = sensor_summary["sensor_mode"]

    rep_summary["report_source"] = build_report_source(sensor_mode)
    rep_summary["last_rep_quality"] = last_rep_quality
    rep_summary["phase_quality"] = flatten_phase_quality(last_rep_quality.get("phase_quality", {}))
    rep_summary["main_issue"] = rule.get("main_issue")
    rep_summary["detected_errors"] = rule.get("detected_errors", [])
    rep_summary["selected_feedback"] = selected_feedback[0] if selected_feedback else None
    rep_summary["selected_feedback_history"] = selected_feedback
    rep_summary["sensor_mode"] = sensor_mode
    rep_summary["sensor_summary"] = sensor_summary
    rep_summary["feedback_trace"] = build_feedback_trace(frame_states)
    return rep_summary


def flatten_phase_quality(phase_quality: dict) -> dict:
    out = {}
    for phase in ("standing", "descent", "bottom", "ascent"):
        item = phase_quality.get(phase, {})
        out[phase] = {
            "confidence": item.get("mean_confidence", 0.0),
            "lower_body_visibility": item.get("mean_lower_body_visibility", 0.0),
            "quality": item.get("level", "bad"),
            "frame_count": item.get("frame_count", 0),
            "has_capture": item.get("has_capture", False),
            "reason": item.get("reason", "-"),
        }
    return out


def build_report_source(sensor_mode: str) -> str:
    if sensor_mode in {"mock", "serial", "jsonl", "http"}:
        return f"phase_captures + {sensor_mode}_sensor_summary"
    return "phase_captures + no_sensor_summary"


def build_sensor_summary(frame_states: list[FrameState]) -> dict:
    sources = [state.sensor.source or "none" for state in frame_states]
    sensor_mode = max(set(sources), key=sources.count) if sources else "none"

    def values(attr: str) -> list[float]:
        out = []
        for state in frame_states:
            value = getattr(state.sensor, attr)
            if value is not None:
                out.append(float(value))
        return out

    def bool_count(attr: str) -> int:
        return sum(1 for state in frame_states if getattr(state.sensor, attr) is True)

    def stats(attr: str) -> dict:
        vals = values(attr)
        return {
            "mean": round(sum(vals) / len(vals), 3) if vals else None,
            "min": round(min(vals), 3) if vals else None,
            "max": round(max(vals), 3) if vals else None,
            "samples": len(vals),
        }

    fsr_types = [state.sensor.fsr_scoring_type for state in frame_states if state.sensor.fsr_scoring_type in {"A", "B"}]
    fsr_scoring_type = max(set(fsr_types), key=fsr_types.count) if fsr_types else "A"

    return {
        "sensor_mode": sensor_mode,
        "frame_count": len(frame_states),
        "source_counts": {source: sources.count(source) for source in sorted(set(sources))},
        "left_pressure_ratio": stats("left_pressure_ratio"),
        "right_pressure_ratio": stats("right_pressure_ratio"),
        "heel_pressure_drop_count": bool_count("heel_pressure_drop"),
        "forefoot_bias_count": bool_count("forefoot_bias"),
        "forefoot_shift_percent": stats("forefoot_shift_percent"),
        "fsr_scoring_type": fsr_scoring_type,
        "fsr_scoring_type_counts": {mode: fsr_types.count(mode) for mode in sorted(set(fsr_types))},
        "imu_pitch": stats("imu_pitch"),
        "imu_roll": stats("imu_roll"),
        "imu_instability": stats("imu_instability"),
        "core_pressure_ratio": stats("core_pressure_ratio"),
    }


def rounded_or_none(value, digits: int = 3):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def build_feedback_trace(frame_states: list[FrameState], limit: int = 12) -> list[dict]:
    """Build a compact explanation trail for live feedback decisions.

    The trace is intentionally summary-level: it records the phase, issue,
    chosen cue, decision reason, and normalized pose/sensor evidence already
    computed by the backend. It does not add raw sensor acquisition.
    """
    trace: list[dict] = []
    last_key: tuple | None = None
    for state in frame_states:
        issue = state.selected_error or ""
        cue = state.realtime_feedback or ""
        if not issue and not cue:
            continue
        key = (state.set_index, state.set_rep_count, state.phase, issue, cue)
        if key == last_key:
            continue
        last_key = key
        sensor_payload = {
            "source": state.sensor.source,
            "left_pressure_ratio": rounded_or_none(state.sensor.left_pressure_ratio),
            "right_pressure_ratio": rounded_or_none(state.sensor.right_pressure_ratio),
            "core_pressure_ratio": rounded_or_none(state.sensor.core_pressure_ratio),
            "imu_pitch": rounded_or_none(state.sensor.imu_pitch),
            "imu_roll": rounded_or_none(state.sensor.imu_roll),
            "heel_pressure_drop": state.sensor.heel_pressure_drop,
            "forefoot_bias": state.sensor.forefoot_bias,
            "forefoot_shift_percent": rounded_or_none(state.sensor.forefoot_shift_percent),
            "fsr_scoring_type": state.sensor.fsr_scoring_type,
        }
        metrics_payload = {
            "avg_knee_angle": rounded_or_none(state.metrics.get("avg_knee_angle")),
            "torso_angle": rounded_or_none(state.metrics.get("torso_angle")),
            "knee_valgus_score": rounded_or_none(state.metrics.get("knee_valgus_score")),
        }
        trace.append({
            "timestamp": rounded_or_none(state.timestamp),
            "frame_id": state.frame_id,
            "set_index": state.set_index,
            "set_rep_count": state.set_rep_count,
            "total_reps": state.total_reps,
            "phase": state.phase,
            "issue": issue or "None",
            "cue": cue,
            "decision_reason": state.decision_reason or "No live cue selected.",
            "detected_errors": list(state.detected_errors or []),
            "metrics": metrics_payload,
            "sensor": sensor_payload,
            "live_feedback": describe_live_feedback(issue, state.phase, cue, sensor_payload, state.decision_reason),
        })
    return trace[-limit:]


def build_set_window_summary(label: str, frame_states: list[FrameState]) -> dict:
    """Summarize one set window from already-normalized pose/sensor state."""
    if not frame_states:
        return {"label": label, "frame_count": 0, "detected_errors": [], "metrics": {}, "sensor_summary": {}}

    all_errors = [error for state in frame_states for error in state.detected_errors]
    feedback_count = sum(1 for state in frame_states if state.realtime_feedback)
    phases = [state.phase for state in frame_states if state.phase]

    def metric_values(key: str) -> list[float]:
        out = []
        for state in frame_states:
            value = state.metrics.get(key)
            if value is not None:
                out.append(float(value))
        return out

    def avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    def peak_abs(values: list[float]) -> float | None:
        return round(max(values, key=abs), 3) if values else None

    metrics = {
        "torso_angle_mean": avg(metric_values("torso_angle")),
        "torso_angle_peak_abs": peak_abs(metric_values("torso_angle")),
        "feedback_count": feedback_count,
    }

    return {
        "label": label,
        "start_time": frame_states[0].timestamp,
        "end_time": frame_states[-1].timestamp,
        "duration_seconds": round(frame_states[-1].timestamp - frame_states[0].timestamp, 2),
        "frame_count": len(frame_states),
        "phase_counts": {phase: phases.count(phase) for phase in sorted(set(phases))},
        "detected_errors": sorted(set(all_errors)),
        "error_count": len(all_errors),
        "metrics": metrics,
        "sensor_summary": build_sensor_summary(frame_states),
    }


def build_set_comparison(set1: dict | None, set2: dict | None) -> dict:
    set1 = set1 or {"label": "SET 1", "frame_count": 0, "metrics": {}, "sensor_summary": {}}
    set2 = set2 or {"label": "SET 2", "frame_count": 0, "metrics": {}, "sensor_summary": {}}

    def metric(name: str):
        return (set1.get("metrics") or {}).get(name), (set2.get("metrics") or {}).get(name)

    def delta(name: str):
        before, after = metric(name)
        if before is None or after is None:
            return None
        return round(float(after) - float(before), 3)

    before_errors = int(set1.get("error_count") or 0)
    after_errors = int(set2.get("error_count") or 0)
    before_feedback = int((set1.get("metrics") or {}).get("feedback_count") or 0)
    after_feedback = int((set2.get("metrics") or {}).get("feedback_count") or 0)

    return {
        "set1": set1,
        "set2": set2,
        "error_count_delta": after_errors - before_errors,
        "feedback_count_delta": after_feedback - before_feedback,
        "torso_angle_peak_abs_delta": delta("torso_angle_peak_abs"),
        "summary": (
            "SET 1/SET 2 comparison uses only Core Bracing, FSR Ground Balance, "
            "heel maintenance, and Spine Alignment rubric evidence."
        ),
    }


def build_set_comparison_summary(before_errors: int, after_errors: int, before_feedback: int, after_feedback: int) -> str:
    if after_errors < before_errors:
        trend = "SET 2 detected fewer rule-based issues than SET 1."
    elif after_errors > before_errors:
        trend = "SET 2 still needs attention; more issues were detected than SET 1."
    else:
        trend = "SET 1 and SET 2 had a similar detected issue count."
    return f"{trend} Live feedback cues: SET 1={before_feedback}, SET 2={after_feedback}."


def build_score_summary(rep_summary: dict, set_comparison: dict | None = None) -> dict:
    """SafeLift 100-point score based on the user-provided rubric.

    Total = Core Bracing 35 + FSR Ground Balance 35 + Spine Alignment 30.
    Existing Unity fields are preserved for compatibility, while the full
    rubric evidence is included under score_summary["rubric"].
    """

    return build_rubric_score_summary(rep_summary, set_comparison)


def build_score_analysis(rep_scores: list[dict]) -> dict:
    """Build the six-rep rubric analysis and expose low/high REP evidence."""

    valid = [
        item for item in (rep_scores or [])
        if isinstance(item, dict) and isinstance(item.get("score"), (int, float))
    ]
    set1 = [item for item in valid if item.get("set") == "SET 1"]
    set2 = [item for item in valid if item.get("set") == "SET 2"]

    def best(items: list[dict]) -> dict | None:
        return max(items, key=lambda item: float(item.get("score", -1)), default=None)

    set1_best = best(set1)
    set2_best = best(set2)
    final_best = best(valid)
    lowest_rep = min(valid, key=lambda item: float(item.get("score", 101)), default=None)
    set1_score = float(set1_best["score"]) if set1_best else None
    set2_score = float(set2_best["score"]) if set2_best else None
    improvement_percent = None
    if set1_score is not None and set2_score is not None and set1_score > 0:
        improvement_percent = round((set2_score - set1_score) / set1_score * 100.0, 1)

    if improvement_percent is not None and improvement_percent > 0:
        improvement_message = (
            f"실시간 피드백이 제공된 SET 2의 점수가 SET 1에 비해 "
            f"{improvement_percent:.1f}% 향상되었습니다."
        )
    elif set1_score is not None and set2_score is not None:
        improvement_message = "실시간 피드백이 제공된 SET 2의 점수가 SET 1에 비해 향상되지 않았습니다."
    else:
        improvement_message = "SET 1과 SET 2의 최고점 비교에 필요한 REP 점수가 아직 부족합니다."

    return {
        "rep_scores": valid,
        "set1_rep_scores": set1,
        "set2_rep_scores": set2,
        "set1_best": set1_best,
        "set2_best": set2_best,
        "final_best": final_best,
        "final_score": final_best.get("score") if final_best else None,
        "final_set": final_best.get("set") if final_best else None,
        "final_rep": final_best.get("rep") if final_best else None,
        "final_grade": (final_best.get("score_summary") or {}).get("grade") if final_best else None,
        "lowest_rep": lowest_rep,
        "highest_rep": final_best,
        "improvement_percent": improvement_percent,
        "improvement_message": improvement_message,
    }


def build_unity_score_summary(payload: dict) -> dict:
    """Adapt an authoritative Unity/Fusion REP score to report JSON."""
    total = round(float(payload.get("total_score") or 0.0), 1)
    fsr = round(float(payload.get("fsr_score") or 0.0), 1)
    fsr_balance = round(float(payload.get("fsr_balance_score") or fsr), 1)
    fsr_heel = round(float(payload.get("fsr_heel_score") or 0.0), 1)
    bracing = round(float(payload.get("bracing_score") or 0.0), 1)
    spine = round(float(payload.get("spine_score") or 0.0), 1)
    grade = "S" if total >= 90 else "A" if total >= 75 else "B" if total >= 60 else "C"
    return {
        "safe_lift_score": total,
        "safe_lift_score_raw": total,
        "balance_score": fsr,
        "core_bracing_score": bracing,
        "spine_alignment_score": spine,
        "torso_stability_score": spine,
        "knee_alignment_score": fsr,
        "max_points": 100,
        "grade": grade,
        "grade_label": "Unity/Fusion authoritative score",
        "basis": "unity_safelift_scoring_v1",
        "score_source": payload.get("source") or "Unity/Fusion SafeLiftScoring",
        "score_evidence_complete": True,
        "score_confidence": "complete",
        "missing_score_evidence": [],
        "rubric": {
            "ground_balance_fsr": {
                "score": fsr,
                "balance_score": fsr_balance,
                "heel_maintain_score": fsr_heel,
            },
            "core_bracing": {"score": bracing},
            "spine_alignment": {"score": spine},
        },
        "summary": f"Unity/Fusion SafeLift score {total:.1f}/100.",
    }


def build_set_improvement_message(score_analysis: dict) -> str:
    """Return the team's required SET 1/SET 2 conclusion sentence."""
    set1 = score_analysis.get("set1_best") or {}
    set2 = score_analysis.get("set2_best") or {}
    try:
        set1_score = float(set1["score"])
        set2_score = float(set2["score"])
    except (KeyError, TypeError, ValueError):
        return "SET 1과 SET 2의 최고점 비교를 위한 REP 점수가 아직 부족합니다."

    if set2_score > set1_score and set1_score > 0:
        improvement = (set2_score - set1_score) / set1_score * 100.0
        return f"실시간 피드백을 제공 받은 Set2가 Set1에 비해 {improvement:.1f}%의 점수 향상률을 보였습니다."
    if set1_score > set2_score:
        return "다음 번엔 실시간 피드백에 집중해서 스쿼트를 해주세요!"
    return "실시간 피드백을 제공 받은 Set2의 점수가 Set1보다 높지 않았습니다."


def main() -> int:
    args = parse_args()
    if args.sensor_http and not args.browser:
        print("--sensor_http requires --browser because /api/sensor_frame is served by the browser backend.")
        return 1

    output_dir = Path(args.output_dir)
    captures_dir = output_dir / "captures"
    reports_dir = output_dir / "reports"
    captures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cap, camera_index = open_camera(args.camera, args.width, args.height, args.fourcc)
    if not cap.isOpened():
        print(f"Camera startup failed for requested index: {args.camera}. No MediaPipe camera window was created.")
        print("Close Unity/Teams/Camera apps that may hold the webcam, then retry with -Camera auto or the correct index.")
        return 1

    side_cap = None
    side_tracker = None
    side_camera_index = -1
    if args.side_camera:
        side_cap, side_camera_index = open_camera(args.side_camera, args.width, args.height, args.fourcc)
        if not side_cap.isOpened():
            print(f"Side camera startup failed for requested index: {args.side_camera}. Continuing with front camera only.")
            side_cap = None
        else:
            side_tracker = MediaPipePoseTracker(args.model_complexity, args.min_detection_confidence, args.min_tracking_confidence)
            print(f"Side camera index: {side_camera_index} (metrics only; no preview window)")

    tracker = MediaPipePoseTracker(args.model_complexity, args.min_detection_confidence, args.min_tracking_confidence)
    phase_detector = PhaseDetector()
    if args.strict_tracking_quality:
        rule_confidence_threshold = args.rule_confidence_threshold if args.rule_confidence_threshold is not None else 0.45
        error_detector = RuleErrorDetector(confidence_threshold=rule_confidence_threshold)
        tracking_quality_gate = TrackingQualityGate(visibility_threshold=0.45, presence_threshold=0.45)
        rep_quality_evaluator = LastRepQualityEvaluator(
            bad_threshold=0.45,
            weak_threshold=0.60,
            min_phase_frames=1,
            require_core_captures=True,
            allow_missing_phases=False,
        )
        print("Tracking quality mode: strict")
    else:
        rule_confidence_threshold = args.rule_confidence_threshold if args.rule_confidence_threshold is not None else 0.15
        error_detector = RuleErrorDetector(confidence_threshold=rule_confidence_threshold)
        tracking_quality_gate = TrackingQualityGate(visibility_threshold=0.15, presence_threshold=0.25)
        rep_quality_evaluator = LastRepQualityEvaluator(
            bad_threshold=0.15,
            weak_threshold=0.45,
            min_phase_frames=1,
            require_core_captures=False,
            allow_missing_phases=True,
        )
        print("Tracking quality mode: prototype-relaxed")
    feedback_engine = RiskAwareFeedbackEngine()
    # The product flow is two sets of three squats: SET 1 practice, rest,
    # SET 2 live feedback, then the final report.
    rep_counter = SquatRepCounter(reps_per_set=3)
    calibration_tracker = CalibrationTracker(duration_seconds=3.0)
    capture_manager = PhaseCaptureManager(str(captures_dir))
    summary_builder = RepSummaryBuilder()
    reporter = make_reporter(args)
    sensor_provider = create_sensor_provider(args)
    avatar_udp = None
    if not args.no_avatar_stream:
        try:
            avatar_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"Unity avatar pose stream: UDP {args.avatar_udp_host}:{args.avatar_udp_port}")
        except OSError as exc:
            print(f"Unity avatar pose stream unavailable: {exc}")

    def send_avatar_pose(landmarks: dict, pose_is_usable: bool, full_body_ready: bool = False) -> None:
        if avatar_udp is None:
            return
        pose = []
        if landmarks:
            for index in range(33):
                point = landmarks.get(index)
                if point is None:
                    pose.append({"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0})
                else:
                    pose.append({
                        "x": float(point.get("x", 0.0)),
                        "y": float(point.get("y", 0.0)),
                        "z": float(point.get("z", 0.0)),
                        "visibility": float(point.get("visibility", 0.0)),
                    })
        packet = {
            "pose": pose,
            "left_hand": [],
            "right_hand": [],
            # Keep transport availability separate from strict full-body
            # trust. Unity can continue upper-body retargeting while waiting
            # for a valid standing calibration.
            "has_visibility": bool(full_body_ready),
            # The avatar stream gate is intentionally more tolerant than the
            # strict rep/scoring gate. Unity's standing-pose sanity check
            # performs the final calibration decision for the leg IK.
            "lower_body_trusted": bool(pose_is_usable),
            "full_body_ready": bool(full_body_ready),
            "reject_reason": "" if pose_is_usable else "front pose lower-body quality gate",
        }
        try:
            avatar_udp.sendto(
                (json.dumps(packet, separators=(",", ":")) + "<EOM>").encode("utf-8"),
                (args.avatar_udp_host, args.avatar_udp_port),
            )
        except OSError:
            pass
    dashboard = DashboardRenderer()
    browser = BrowserDashboard(
        args.web_host,
        args.web_port,
        sensor_push_provider=sensor_provider,
        static_report_root=args.report_share_root or None,
        public_report_base=args.report_public_base,
    ) if args.browser else None
    if browser:
        browser.start()

    rep_id = 1
    frame_id = 0
    rep_states: list[FrameState] = []
    session_files: set[Path] = set()
    latest_report = ""
    latest_report_id = ""
    latest_last_rep_quality: dict | None = None
    latest_rep_summary: dict | None = None
    latest_report_source = "-"
    latest_command_status = "ready"
    latest_report_timing: dict | None = None
    latest_phase_images: dict | None = None
    latest_report_status = "none"
    latest_report_started_at: float | None = None
    latest_report_timeout_seconds = 10.0
    latest_report_message = "Report has not started."
    latest_report_fallback = False
    set1_baseline_summary: dict | None = None
    rep_score_history: list[dict] = []
    current_rep_states: list[FrameState] = []
    auto_report_started = False
    prev_hip_height = None
    prev_timestamp = None
    fps = 0.0
    last_time = time.time()
    # Keep the camera in its natural orientation. The previous default
    # mirrored the feed like a selfie camera.
    flip = False

    print("SafeLift XR backend started.")
    print("Keys: q/Esc quit, r reset session, s save snapshot JSON.")
    print("Report generation is locked until SET 1 three reps + SET 2 three reps are completed by the squat counter.")
    print(f"Camera index: {camera_index}")

    def reset_current_rep() -> None:
        nonlocal latest_report, latest_report_id, latest_phase_images, latest_last_rep_quality, latest_rep_summary, latest_report_source, latest_command_status, latest_report_timing, latest_report_status, latest_report_started_at, latest_report_message, latest_report_fallback, set1_baseline_summary, rep_score_history, current_rep_states, auto_report_started
        phase_detector.reset()
        rep_counter.reset()
        calibration_tracker.reset()
        feedback_engine.reset_rep()
        capture_manager.reset()
        rep_states.clear()
        latest_report = ""
        latest_phase_images = None
        latest_last_rep_quality = None
        latest_rep_summary = None
        latest_report_source = "-"
        latest_report_timing = None
        latest_report_status = "none"
        latest_report_started_at = None
        latest_report_message = "Report has not started."
        latest_report_fallback = False
        set1_baseline_summary = None
        rep_score_history.clear()
        if browser:
            browser.clear_unity_rep_scores()
        current_rep_states.clear()
        auto_report_started = False
        latest_command_status = "current rep reset"
        print("Current rep reset.")

    def save_snapshot(state: FrameState) -> None:
        nonlocal latest_command_status
        snapshot_path = reports_dir / f"frame_{state.frame_id}_state.json"
        snapshot_path.write_text(json.dumps(frame_state_to_jsonable(state), ensure_ascii=False, indent=2), encoding="utf-8")
        session_files.add(snapshot_path)
        latest_command_status = f"snapshot saved: frame {state.frame_id}"
        print(f"Saved state snapshot: {snapshot_path}")

    def publish_browser_state(command_status: str | None = None) -> None:
        if not browser or not rep_states:
            return
        if command_status is not None:
            nonlocal_latest_status[0] = command_status
        state = rep_states[-1]
        live_tracking_quality = tracking_quality_gate.evaluate([state], {})
        browser_state = frame_state_to_jsonable(state)
        browser_state["latest_report"] = latest_report
        browser_state["latest_report_id"] = latest_report_id
        browser_state["latest_rep_summary"] = latest_rep_summary
        browser_state["live_tracking_quality"] = live_tracking_quality
        browser_state["last_rep_quality"] = latest_last_rep_quality
        browser_state["report_source"] = latest_report_source
        browser_state["command_status"] = nonlocal_latest_status[0]
        browser_state["report_timing"] = latest_report_timing
        browser_state["report_status"] = latest_report_status
        browser_state["report_started_at"] = latest_report_started_at
        browser_state["report_elapsed_seconds"] = round(time.time() - latest_report_started_at, 2) if latest_report_started_at else 0.0
        browser_state["report_timeout_seconds"] = latest_report_timeout_seconds
        browser_state["report_timeout"] = bool(latest_report_started_at and (time.time() - latest_report_started_at) >= latest_report_timeout_seconds and latest_report_status == "generating")
        browser_state["report_message"] = latest_report_message
        browser_state["report_fallback"] = latest_report_fallback
        browser_state["tracking_quality"] = live_tracking_quality
        browser.update_state(browser_state)

    nonlocal_latest_status = [latest_command_status]

    def record_completed_rep_score(frame_window: list[FrameState]) -> None:
        if not frame_window:
            return
        last_state = frame_window[-1]
        # Unity/Fusion owns the user-facing score. Infer the set from the
        # ordered completed REP history when Python's set_index is stale.
        completed_index = len(rep_score_history)
        reps_per_set = max(1, int(rep_counter.reps_per_set))
        inferred_set_number = completed_index // reps_per_set + 1
        inferred_rep_number = completed_index % reps_per_set + 1
        reported_set_number = int(last_state.set_index or 1)
        reported_rep_number = int(last_state.set_rep_count or 0)
        if reported_set_number >= 2 and reported_rep_number > 0:
            set_label = "SET 2"
            rep_number = min(reps_per_set, reported_rep_number)
        elif inferred_set_number >= 2:
            set_label = "SET 2"
            rep_number = inferred_rep_number
        else:
            set_label = "SET 1"
            rep_number = reported_rep_number if reported_rep_number > 0 else inferred_rep_number
        summary = summary_builder.build(
            rep_id * 10 + rep_number,
            frame_window,
            {},
        )
        summary["calibration_baseline"] = last_state.calibration_baseline or {}
        quality = rep_quality_evaluator.evaluate(frame_window, {})
        summary = enrich_rep_summary_for_vlm(summary, frame_window, quality)
        unity_score = browser.get_unity_rep_score(set_label, rep_number) if browser else None
        score_summary = build_unity_score_summary(unity_score) if unity_score else build_score_summary(summary, None)
        rep_score_history.append({
            "set": set_label,
            "rep": rep_number,
            "score": score_summary.get("safe_lift_score_raw", score_summary.get("safe_lift_score")),
            "score_summary": score_summary,
            "deductions": {
                "fsr_balance": round(max(0, 35 - score_summary.get("balance_score", 0)), 1),
                "heel_maintenance": round(max(0, 12 - ((score_summary.get("rubric") or {}).get("ground_balance_fsr") or {}).get("heel_maintain_score", 0)), 1),
                "spine_alignment": round(max(0, 30 - score_summary.get("spine_alignment_score", 0)), 1),
                "core_bracing": round(max(0, 35 - score_summary.get("core_bracing_score", 0)), 1),
            },
            "duration_seconds": round(last_state.timestamp - frame_window[0].timestamp, 2),
            "score_source": score_summary.get("score_source") or "Python report fallback",
        })

    def wait_for_unity_scores(timeout_seconds: float = 5.0) -> int:
        """Give Unity time to upload all six authoritative REP scores."""
        if not browser:
            return 0
        expected = [(f"SET {set_number}", rep) for set_number in (1, 2) for rep in (1, 2, 3)]
        deadline = time.time() + max(0.0, timeout_seconds)
        while time.time() < deadline:
            received = sum(1 for set_label, rep in expected if browser.get_unity_rep_score(set_label, rep))
            if received == len(expected):
                return received
            time.sleep(0.05)
        return sum(1 for set_label, rep in expected if browser.get_unity_rep_score(set_label, rep))

    def refresh_authoritative_rep_scores() -> int:
        """Replace any early Python fallback with late Unity REP packets."""
        if not browser:
            return 0
        refreshed = 0
        for item in rep_score_history:
            unity_score = browser.get_unity_rep_score(str(item.get("set")), int(item.get("rep") or 0))
            if not unity_score:
                continue
            score_summary = build_unity_score_summary(unity_score)
            item["score"] = score_summary["safe_lift_score_raw"]
            item["score_summary"] = score_summary
            item["deductions"] = {
                "fsr_balance": round(max(0.0, 35.0 - score_summary.get("balance_score", 0.0)), 1),
                "heel_maintenance": round(max(0.0, 15.0 - ((score_summary.get("rubric") or {}).get("ground_balance_fsr") or {}).get("heel_maintain_score", 0.0)), 1),
                "spine_alignment": round(max(0.0, 30.0 - score_summary.get("spine_alignment_score", 0.0)), 1),
                "core_bracing": round(max(0.0, 35.0 - score_summary.get("core_bracing_score", 0.0)), 1),
            }
            item["score_source"] = score_summary.get("score_source")
            refreshed += 1
        return refreshed

    def finish_rep(force: bool = False) -> None:
        nonlocal rep_id, latest_report, latest_report_id, latest_phase_images, latest_last_rep_quality, latest_rep_summary, latest_report_source, latest_command_status, latest_report_timing, latest_report_status, latest_report_started_at, latest_report_message, latest_report_fallback, set1_baseline_summary
        required_reps = rep_counter.reps_per_set * 2
        # Unity can use the Fusion Monitor as the authoritative per-set
        # counter. In that mode SET 2 completion is already confirmed by the
        # user flow, so do not block report generation on the backend's own
        # MediaPipe counter reaching 6 a second time.
        if not force and rep_counter.total_reps < required_reps:
            latest_command_status = (
                f"report locked: complete SET 1 and SET 2 first "
                f"({rep_counter.total_reps}/{required_reps} reps)"
            )
            latest_report_status = "none"
            latest_report_started_at = None
            latest_report_message = "SET 1 three reps and SET 2 three reps must be completed before the final report can be generated."
            nonlocal_latest_status[0] = latest_command_status
            publish_browser_state(latest_command_status)
            print(latest_command_status)
            return
        if set1_baseline_summary is None and not force:
            latest_command_status = "report locked: SET 1 baseline is missing; restart the session and complete SET 1 first."
            latest_report_status = "none"
            latest_report_started_at = None
            latest_report_message = "SET 1 baseline is required before SET 2 comparison and final scoring."
            nonlocal_latest_status[0] = latest_command_status
            publish_browser_state(latest_command_status)
            print(latest_command_status)
            return
        if set1_baseline_summary is None and force:
            # Fusion has already confirmed that the user completed both
            # sets. Keep report generation available even when the separate
            # MediaPipe backend did not publish its SET 1 boundary; the
            # comparison simply reports that the baseline evidence is absent.
            set1_baseline_summary = build_set_window_summary("SET 1", [])
            latest_command_status = "forced report: SET 2 completed by Fusion; SET 1 baseline unavailable"
            nonlocal_latest_status[0] = latest_command_status
        if len(rep_states) < 5:
            latest_command_status = f"not enough frames: {len(rep_states)}"
            nonlocal_latest_status[0] = latest_command_status
            print("Not enough frames for a report yet.")
            return
        report_start = time.perf_counter()
        latest_report_started_at = time.time()
        latest_report_status = "generating"
        latest_report_fallback = False
        latest_report_message = "AI Coach Report is being generated. If it takes more than 10 seconds, Unity will show the basic rule-based analysis first."
        latest_report_timing = None
        latest_command_status = (
            f"분석 중: rep {rep_id}. 첫 분석은 약 15~20초, 이후에는 약 8~10초 정도 걸립니다. "
            "그동안 다음 반복 자세를 준비하세요."
        )
        latest_report = (
            "[분석 중]\n"
            "AI가 방금 반복의 주요 장면과 센서 요약을 보고 있습니다.\n"
            "첫 분석은 약 15~20초, 이후 분석은 보통 약 8~10초 걸립니다.\n\n"
            "기다리는 동안 이것만 확인하세요.\n"
            "1. 다음 반복 시작 위치에서 골반, 무릎, 발목, 발끝이 화면에 들어오는지 확인하세요.\n"
            "2. 방금 반복에서 무릎이 안쪽으로 모였는지 떠올리고, 다음에는 발끝 방향으로 밀 준비를 하세요.\n"
            "3. 발바닥 전체가 바닥에 붙은 상태에서 천천히 시작하세요."
        )
        nonlocal_latest_status[0] = latest_command_status
        publish_browser_state(latest_command_status)

        capture_start = time.perf_counter()
        phase_snapshots = capture_manager.finalize_rep(rep_id)
        capture_elapsed = time.perf_counter() - capture_start
        session_files.update(Path(path) for path in phase_snapshots.values())
        unity_scores_received = wait_for_unity_scores(timeout_seconds=5.0)
        unity_scores_refreshed = refresh_authoritative_rep_scores()
        summary_start = time.perf_counter()
        rep_summary = summary_builder.build(rep_id, rep_states, phase_snapshots)
        last_rep_quality = rep_quality_evaluator.evaluate(rep_states, phase_snapshots)
        summary_elapsed = time.perf_counter() - summary_start
        if not last_rep_quality["report_allowed"]:
            rep_summary["rule_analysis"]["main_issue"] = "TrackingUncertain"
            rep_summary["rule_analysis"]["detected_errors"] = ["TrackingUncertain"]
            rep_summary["rule_analysis"]["deferred_errors"] = []
            rep_summary["rule_analysis"]["selected_realtime_feedback"] = [
                "자세 추적이 불안정합니다. 전신이 화면에 들어오도록 카메라 위치를 조정해주세요."
            ]
        rep_summary = enrich_rep_summary_for_vlm(rep_summary, rep_states, last_rep_quality)
        set2_summary = build_set_window_summary("SET 2", rep_states)
        rep_summary["set_comparison"] = build_set_comparison(set1_baseline_summary, set2_summary)
        rep_summary["score_analysis"] = build_score_analysis(rep_score_history)
        rep_summary["score_analysis"]["improvement_message"] = build_set_improvement_message(rep_summary["score_analysis"])
        rep_summary["score_analysis"]["unity_scores_received"] = unity_scores_received
        rep_summary["score_analysis"]["unity_scores_refreshed"] = unity_scores_refreshed
        final_best = rep_summary["score_analysis"].get("final_best")
        rep_summary["score_summary"] = (
            dict(final_best.get("score_summary") or {})
            if final_best
            else build_score_summary(rep_summary, rep_summary["set_comparison"])
        )
        rep_summary["score_summary"]["score_analysis"] = rep_summary["score_analysis"]
        summary_path = reports_dir / f"rep{rep_id}_summary.json"
        report_path = reports_dir / f"rep{rep_id}_report.txt"
        summary_builder.save(rep_summary, str(summary_path))
        session_files.add(summary_path)
        print("Generating report with Gemini (template fallback if unavailable)...")
        ai_start = time.perf_counter()
        coach_report = reporter.generate(rep_summary)
        improvement_message = rep_summary["score_analysis"]["improvement_message"]
        if improvement_message not in coach_report:
            coach_report = coach_report.rstrip() + "\n\n[SET 1/SET 2 핵심 결과]\n" + improvement_message
        ai_elapsed = time.perf_counter() - ai_start
        total_elapsed = time.perf_counter() - report_start
        reporter_status = reporter.status() if hasattr(reporter, "status") else ""
        latest_report_fallback = "fallback" in reporter_status.lower() or "template" in reporter_status.lower()
        latest_report_status = "ready"
        latest_report_message = (
            "Template fallback report is ready."
            if latest_report_fallback
            else "Gemini report is ready."
        )
        latest_report_timing = {
            "total_seconds": round(total_elapsed, 2),
            "capture_seconds": round(capture_elapsed, 2),
            "summary_seconds": round(summary_elapsed, 2),
            "ai_seconds": round(ai_elapsed, 2),
            "timeout_seconds": latest_report_timeout_seconds,
            "timed_out": total_elapsed >= latest_report_timeout_seconds,
            "fallback": latest_report_fallback,
        }
        rep_summary["report_timing"] = latest_report_timing
        latest_report = coach_report
        latest_last_rep_quality = last_rep_quality
        latest_rep_summary = rep_summary
        latest_report_source = rep_summary["report_source"]
        report_path.write_text(latest_report, encoding="utf-8")
        session_files.add(report_path)
        latest_phase_images = load_phase_images(phase_snapshots)
        if browser:
            # Final snapshots are created after the camera loop's normal
            # browser.update() call. Encode them explicitly before publishing
            # the archived/static report, otherwise the HTML has broken image
            # links even though the JPEG files were captured successfully.
            browser.update_phase_images(latest_phase_images)
        print(f"Saved rep summary: {summary_path}")
        print(f"Saved report: {report_path}")
        latest_command_status = f"report saved: rep {rep_id} ({latest_report_timing['total_seconds']:.2f}s)"
        nonlocal_latest_status[0] = latest_command_status
        publish_browser_state(latest_command_status)
        if hasattr(reporter, "status"):
            print(reporter.status())
        rep_id += 1
        rep_states.clear()
        capture_manager.reset()
        feedback_engine.reset_rep()

    try:
        if not args.no_camera_window:
            cv2.namedWindow("SafeLift MediaPipe Camera", cv2.WINDOW_NORMAL)
            preview_width = 640 if args.camera_half != "both" else 1280
            cv2.resizeWindow("SafeLift MediaPipe Camera", preview_width, 480)
        while True:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: Could not read camera frame.")
                break
            if args.camera_half != "both":
                split = frame.shape[1] // 2
                frame = frame[:, :split].copy() if args.camera_half == "left" else frame[:, split:].copy()
                frame = trim_dark_padding(frame)
                # Normalize the selected OBS half after removing its black
                # letterbox padding.
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            # Select the requested half before mirroring. Mirroring the full
            # composite first would swap the left/right OBS sources.
            if flip:
                frame = cv2.flip(frame, 1)

            now = time.time()
            dt = now - last_time
            last_time = now
            fps = (0.9 * fps + 0.1 * (1.0 / dt)) if dt > 0 and fps > 0 else (1.0 / dt if dt > 0 else 0.0)
            frame_id += 1

            landmarks, confidence, annotated = tracker.process_frame(frame)
            # MediaPipe may return an annotated copy using the capture
            # driver's original dimensions. Keep every visible output at the
            # selected camera-half's native 640x480 ratio.
            if args.camera_half != "both" and annotated is not None and annotated.size:
                annotated = cv2.resize(annotated, (640, 480), interpolation=cv2.INTER_AREA)
            metrics = compute_pose_metrics(landmarks, prev_hip_height, now - prev_timestamp if prev_timestamp else None)
            prev_hip_height = metrics.get("hip_height") if metrics.get("hip_height") is not None else prev_hip_height
            prev_timestamp = now
            pose_is_usable = tracking_quality_gate.pose_is_usable(landmarks, confidence)
            avatar_pose_is_sendable = tracking_quality_gate.pose_is_sendable(landmarks, confidence)
            metrics["pose_frame_usable"] = pose_is_usable
            phase = "Unknown"
            counter_metrics = metrics
            # Keep squat scoring/calibration strict, but retain the original
            # GIST_AVATA behavior for avatar streaming: a mildly weak ankle
            # frame should not disconnect the avatar from the user's motion.
            send_avatar_pose(landmarks, avatar_pose_is_sendable, pose_is_usable)

            # The front camera remains the authoritative rep/phase source.
            # The side camera is sampled only for sagittal posture evidence and
            # is intentionally never sent to the UI or opened in a preview.
            if side_cap is not None and side_tracker is not None:
                side_ok, side_frame = side_cap.read()
                if side_ok and side_frame is not None and side_frame.size:
                    if flip:
                        side_frame = cv2.flip(side_frame, 1)
                    side_landmarks, side_confidence, _ = side_tracker.process_frame(side_frame)
                    side_metrics = compute_pose_metrics(side_landmarks)
                    metrics["side_torso_angle"] = side_metrics.get("torso_angle")
                    metrics["side_avg_knee_angle"] = side_metrics.get("avg_knee_angle")
                    metrics["side_confidence"] = side_confidence
                    metrics["side_pose_frame_usable"] = tracking_quality_gate.pose_is_usable(side_landmarks, side_confidence)
                    if metrics["side_pose_frame_usable"]:
                        # Squat phase and repetition counting use the side view.
                        # Front-view metrics remain available for avatar/UI data.
                        counter_metrics = dict(side_metrics)
                        counter_metrics["pose_frame_usable"] = True
                        phase = phase_detector.update(counter_metrics)
                else:
                    metrics["side_torso_angle"] = None
                    metrics["side_avg_knee_angle"] = None
                    metrics["side_confidence"] = 0.0
                    metrics["side_pose_frame_usable"] = False

            if phase == "Unknown" and pose_is_usable and side_cap is None:
                phase = phase_detector.update(metrics)

            sensor = sensor_provider.read()

            state = FrameState(
                timestamp=now,
                frame_id=frame_id,
                phase=phase,
                confidence=confidence,
                landmarks=landmarks,
                metrics=metrics,
                sensor=sensor,
            )
            calibration = calibration_tracker.update(now, confidence, sensor)
            state.calibration_status = calibration["status"]
            state.calibration_progress = calibration["progress"]
            state.calibration_message = calibration["message"]
            state.calibration_baseline = calibration["baseline"]
            live_tracking_quality = tracking_quality_gate.evaluate([state], {})
            rep_snapshot = rep_counter.update(phase, live_tracking_quality, metrics=counter_metrics, timestamp=now)
            state.total_reps = rep_snapshot.total_reps
            state.set_index = rep_snapshot.set_index
            state.set_rep_count = rep_snapshot.set_rep_count
            state.rep_completed = rep_snapshot.last_completed
            state.detected_errors = error_detector.detect(state)
            decision = feedback_engine.select_feedback(state.phase, state.detected_errors, state.timestamp)
            state.selected_error = decision["selected_error"]
            state.realtime_feedback = decision["feedback"]
            state.deferred_errors = decision["deferred_errors"]
            state.decision_reason = decision["decision_reason"]

            rep_states.append(state)
            current_rep_states.append(state)
            capture_manager.update(annotated, state)

            dashboard_frame = dashboard.render(annotated, state, fps, latest_report, latest_phase_images)
            if browser:
                browser_state = frame_state_to_jsonable(state)
                browser_state["latest_report"] = latest_report
                browser_state["latest_report_id"] = latest_report_id
                browser_state["latest_rep_summary"] = latest_rep_summary
                browser_state["live_tracking_quality"] = live_tracking_quality
                browser_state["last_rep_quality"] = latest_last_rep_quality
                browser_state["report_source"] = latest_report_source
                browser_state["command_status"] = latest_command_status
                browser_state["report_timing"] = latest_report_timing
                browser_state["report_status"] = latest_report_status
                browser_state["report_started_at"] = latest_report_started_at
                browser_state["report_elapsed_seconds"] = round(time.time() - latest_report_started_at, 2) if latest_report_started_at else 0.0
                browser_state["report_timeout_seconds"] = latest_report_timeout_seconds
                browser_state["report_timeout"] = bool(latest_report_started_at and (time.time() - latest_report_started_at) >= latest_report_timeout_seconds and latest_report_status == "generating")
                browser_state["report_message"] = latest_report_message
                browser_state["report_fallback"] = latest_report_fallback
                browser_state["tracking_quality"] = live_tracking_quality
                browser.update(dashboard_frame, browser_state, latest_phase_images)
                browser.update_live_frame(annotated)
            if state.rep_completed:
                # Store one score-analysis item per completed REP. The old
                # boundary-only logic stored SET 1 and SET 2 as two aggregate
                # entries, so the six-REP report could not render REP 1-3 for
                # both sets.
                record_completed_rep_score(current_rep_states)
                current_rep_states.clear()

            if state.rep_completed and state.total_reps == rep_counter.reps_per_set:
                # End SET 1 and start a clean SET 2 baseline while keeping
                # the total count visible to Unity.
                set1_baseline_summary = build_set_window_summary("SET 1", rep_states)
                rep_states.clear()
                capture_manager.reset()
                phase_detector.reset()
            if state.rep_completed and state.total_reps == rep_counter.reps_per_set * 2 and not auto_report_started:
                auto_report_started = True
                finish_rep()
            if not args.no_camera_window:
                # Keep a dedicated pose-evidence window separate from the
                # dashboard so the operator can immediately verify whether
                # the full body, hips, and both knees are actually visible.
                # Show the raw capture as the base; MediaPipe landmarks are
                # drawn on top when available. This prevents a failed pose
                # result from making a valid camera image look like a black
                # preview.
                camera_preview = annotated if annotated is not None and annotated.size else frame
                cv2.imshow("SafeLift MediaPipe Camera", camera_preview)
            cv2.imshow("SafeLift XR", dashboard_frame)
            key = cv2.waitKey(1) & 0xFF
            browser_command = browser.pop_command() if browser else None

            if key in (ord("q"), 27):
                break
            if key == ord("r") or browser_command == "reset_rep":
                reset_current_rep()
            if key == ord("s") or browser_command == "save_snapshot":
                save_snapshot(state)
            if browser_command == "finish_rep":
                finish_rep(force=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        tracker.close()
        if side_cap is not None:
            side_cap.release()
        if side_tracker is not None:
            side_tracker.close()
        if browser:
            browser.stop()
        if hasattr(sensor_provider, "close"):
            sensor_provider.close()
        if avatar_udp is not None:
            avatar_udp.close()
        cv2.destroyAllWindows()
        if args.keep_outputs:
            print(f"Keeping {len(session_files)} output file(s).")
        else:
            deleted = cleanup_session_files(session_files, output_dir)
            print(f"Deleted {deleted} temporary output file(s). Use --keep_outputs to preserve reports/captures.")

    return 0


def run_mock_session(args: argparse.Namespace) -> int:
    """Serve Unity-compatible endpoints without camera or physical sensors.

    This is not the hardware path. It is a deterministic backend simulator so
    the Unity REAL mode can be tested from a fresh clone even when no webcam,
    Quest, or Arduino/SafeLift sensor is attached.
    """
    browser = BrowserDashboard(args.web_host, args.web_port)
    browser.start()
    phases = ["Standing", "Descent", "Bottom", "Ascent", "Standing"]
    scenarios = [
        ("None", [], ""),
        ("KneeValgus", ["KneeValgus"], "무릎을 발끝 방향으로 밀어주세요."),
        ("WeightShift", ["WeightShift"], "좌우 발바닥 압력을 가운데로 맞춰주세요."),
        ("CorePressureDrop", ["CorePressureDrop"], "내려가기 전에 복압을 다시 잡아주세요."),
        ("TorsoLean", ["TorsoLean"], "상체를 조금 더 안정적으로 세워주세요."),
    ]
    reps_per_set = 3
    total_reps = 0
    phase_index = 0
    frame_id = 0
    started = time.time()
    tick_seconds = max(0.01, float(args.mock_tick_seconds))
    phase_frames = max(1, int(args.mock_phase_frames))
    print(f"SafeLift mock backend: http://{args.web_host}:{args.web_port}")
    print("Press Ctrl+C to stop. Unity REAL mode can connect to /api/session_state.")
    try:
        while True:
            now = time.time()
            frame_id += 1
            browser_command = browser.pop_command()
            if browser_command in {"reset_rep", "reset_session"}:
                # Keep the simulator's command contract identical to the
                # camera-backed path. Unity sends reset_rep at the beginning
                # of every Real Mode session and waits for a zero-rep state.
                total_reps = 0
                phase_index = 0
                frame_id = 0
                started = now
            phase = phases[phase_index % len(phases)]
            rep_completed = False
            if phase_index % len(phases) == len(phases) - 1:
                total_reps = min(reps_per_set * 2, total_reps + 1)
                rep_completed = True
            set_index = 1 if total_reps < reps_per_set else 2
            set_rep_count = total_reps if set_index == 1 else max(0, total_reps - reps_per_set)
            if total_reps == reps_per_set:
                set_rep_count = reps_per_set

            main_issue, detected_errors, cue = scenarios[(max(0, total_reps - 1)) % len(scenarios)]
            sensor_payload = {
                "source": "mock",
                "left_pressure_ratio": 0.56 if main_issue == "WeightShift" else 0.50,
                "right_pressure_ratio": 0.44 if main_issue == "WeightShift" else 0.50,
                "imu_pitch": 8.0,
                "imu_roll": 2.0,
                "core_pressure_ratio": 0.42 if main_issue == "CorePressureDrop" else 0.78,
            }
            live_feedback = describe_live_feedback(main_issue, phase, cue, sensor_payload, "mock_session deterministic scenario")
            feedback_trace = build_mock_feedback_trace(main_issue, detected_errors, cue, sensor_payload, phase, set_index, set_rep_count, total_reps, frame_id)
            report_ready = total_reps >= reps_per_set * 2
            report_elapsed = max(0.0, now - started - 8.0) if report_ready else 0.0
            calibration_progress = min(1.0, (now - started) / 3.0)
            calibration_status = "ready" if calibration_progress >= 1.0 else "measuring"
            latest_report = build_mock_session_report(main_issue, detected_errors, cue) if report_ready else ""
            latest_report_id = "mock_session_final" if report_ready else ""
            phase_snapshots = {
                "standing": "/api/phase_capture/standing.jpg",
                "descent": "/api/phase_capture/descent.jpg",
                "bottom": "/api/phase_capture/bottom.jpg",
                "ascent": "/api/phase_capture/ascent.jpg",
            }
            sensor_summary = {
                "sensor_mode": "mock",
                "left_pressure_ratio": {"mean": 0.56 if main_issue == "WeightShift" else 0.50},
                "right_pressure_ratio": {"mean": 0.44 if main_issue == "WeightShift" else 0.50},
                "core_pressure_ratio": {"mean": 0.42 if main_issue == "CorePressureDrop" else 0.78},
                "imu_pitch": {"mean": 8.0},
                "imu_roll": {"mean": 2.0},
            }

            state = {
                "timestamp": now,
                "frame_id": frame_id,
                "phase": phase,
                "confidence": 0.92,
                "metrics": {
                    "avg_knee_angle": 172 if phase == "Standing" else 118 if phase == "Bottom" else 145,
                    "torso_angle": 8.0,
                    "knee_valgus_score": 0.12 if main_issue == "KneeValgus" else 0.02,
                },
                "sensor": sensor_payload,
                "detected_errors": detected_errors,
                "realtime_feedback": cue,
                "selected_error": main_issue,
                "deferred_errors": [],
                "decision_reason": "mock_session deterministic scenario",
                "live_feedback": live_feedback,
                "rep_count": total_reps,
                "set_index": set_index,
                "set_rep_count": set_rep_count,
                "rep_completed": rep_completed,
                "calibration_status": calibration_status,
                "calibration_progress": round(calibration_progress, 3),
                "calibration_message": "Mock baseline ready." if calibration_status == "ready" else "Stand still while mock baseline is measured.",
                "calibration_baseline": {
                    "sensor_mode": "mock",
                    "sample_count": frame_id,
                    "pose_confidence_mean": 0.92,
                    "left_pressure_ratio": {"mean": sensor_payload["left_pressure_ratio"], "min": sensor_payload["left_pressure_ratio"], "max": sensor_payload["left_pressure_ratio"], "samples": frame_id},
                    "right_pressure_ratio": {"mean": sensor_payload["right_pressure_ratio"], "min": sensor_payload["right_pressure_ratio"], "max": sensor_payload["right_pressure_ratio"], "samples": frame_id},
                    "core_pressure_ratio": {"mean": sensor_payload["core_pressure_ratio"], "min": sensor_payload["core_pressure_ratio"], "max": sensor_payload["core_pressure_ratio"], "samples": frame_id},
                    "imu_pitch": {"mean": sensor_payload["imu_pitch"], "min": sensor_payload["imu_pitch"], "max": sensor_payload["imu_pitch"], "samples": frame_id},
                    "imu_roll": {"mean": sensor_payload["imu_roll"], "min": sensor_payload["imu_roll"], "max": sensor_payload["imu_roll"], "samples": frame_id},
                },
                "latest_report": latest_report,
                "latest_report_id": latest_report_id,
                "report_status": "ready" if report_ready else "none",
                "report_started_at": now - report_elapsed if report_ready else None,
                "report_elapsed_seconds": round(report_elapsed, 2),
                "report_timeout_seconds": 10.0,
                "report_timeout": False,
                "report_message": "Mock template report is ready." if report_ready else "Report has not started.",
                "report_fallback": True if report_ready else False,
                "latest_rep_summary": {
                    "rep_id": total_reps,
                    "sensor_mode": "mock",
                    "main_issue": main_issue,
                    "detected_errors": detected_errors,
                    "selected_feedback": cue or None,
                    "selected_feedback_history": [cue] if cue else [],
                    "feedback_trace": feedback_trace,
                    "report_id": latest_report_id,
                    "report_source": "mock_session + template_report",
                    "phase_snapshots": phase_snapshots,
                    "phase_quality": {
                        "standing": {"quality": "good", "frame_count": 8, "has_capture": True, "reason": "mock standing snapshot"},
                        "descent": {"quality": "good", "frame_count": 8, "has_capture": True, "reason": "mock descent snapshot"},
                        "bottom": {"quality": "good", "frame_count": 8, "has_capture": True, "reason": "mock bottom snapshot"},
                        "ascent": {"quality": "good", "frame_count": 8, "has_capture": True, "reason": "mock ascent snapshot"},
                    },
                    "metrics": {
                        "safe_lift_score": 86,
                        "knee_alignment_score": 84,
                        "balance_score": 88,
                        "torso_stability_score": 87,
                        "core_bracing_score": 82,
                    },
                    "score_summary": {
                        "safe_lift_score": 86,
                        "knee_alignment_score": 84,
                        "balance_score": 88,
                        "torso_stability_score": 87,
                        "core_bracing_score": 82,
                        "improvement_delta": 8,
                        "basis": "safelift_rubric_score_v1",
                        "summary": "Mock prototype score payload for no-sensor end-to-end verification.",
                    },
                    "sensor_summary": sensor_summary,
                    "set_comparison": {
                        "set1": {"label": "SET 1", "frame_count": 60, "error_count": 4, "metrics": {"feedback_count": 0, "torso_angle_peak_abs": 10.0, "knee_valgus_peak": 0.18}},
                        "set2": {"label": "SET 2", "frame_count": 60, "error_count": 2, "metrics": {"feedback_count": 2, "torso_angle_peak_abs": 7.5, "knee_valgus_peak": 0.10}},
                        "error_count_delta": -2,
                        "feedback_count_delta": 2,
                        "torso_angle_peak_abs_delta": -2.5,
                        "knee_valgus_peak_delta": -0.08,
                        "summary": "SET 2 detected fewer rule-based issues than SET 1. Live feedback cues: SET 1=0, SET 2=2.",
                    },
                },
                "live_tracking_quality": "GOOD",
                "last_rep_quality": "GOOD",
                "report_source": "mock_session + template_report",
                "command_status": "report ready" if report_ready else "mock backend running",
            }
            frame = np.full((720, 1280, 3), (18, 28, 42), dtype=np.uint8)
            cv2.putText(frame, "SafeLift XR MOCK BACKEND", (42, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (230, 210, 80), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Phase: {phase}", (42, 162), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Rep: {total_reps}/10  Set: {set_index}", (42, 222), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 210, 90), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Issue: {main_issue}", (42, 282), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 220, 220), 2, cv2.LINE_AA)
            browser.update_state(state)
            browser.update_live_frame(frame)
            if frame_id % phase_frames == 0:
                phase_index += 1
            time.sleep(tick_seconds)
    except KeyboardInterrupt:
        print("\nMock backend stopped.")
    finally:
        browser.stop()
    return 0


def build_mock_feedback_trace(
    main_issue: str,
    detected_errors: list[str],
    cue: str,
    sensor_payload: dict,
    phase: str,
    set_index: int,
    set_rep_count: int,
    total_reps: int,
    frame_id: int,
) -> list[dict]:
    if not cue and (not main_issue or main_issue == "None"):
        return []
    issue = main_issue or "None"
    return [
        {
            "timestamp": rounded_or_none(time.time()),
            "frame_id": frame_id,
            "set_index": set_index,
            "set_rep_count": set_rep_count,
            "total_reps": total_reps,
            "phase": phase,
            "issue": issue,
            "cue": cue,
            "decision_reason": "mock_session deterministic scenario",
            "detected_errors": detected_errors,
            "metrics": {
                "avg_knee_angle": 172 if phase == "Standing" else 118 if phase == "Bottom" else 145,
                "torso_angle": 8.0,
                "knee_valgus_score": 0.12 if issue == "KneeValgus" else 0.02,
            },
            "sensor": dict(sensor_payload),
            "live_feedback": describe_live_feedback(issue, phase, cue, sensor_payload, "mock_session deterministic scenario"),
        }
    ]


def build_mock_session_report(main_issue: str, detected_errors: list[str], cue: str) -> str:
    issue_text = main_issue if main_issue and main_issue != "None" else "큰 오류 없음"
    detected_text = ", ".join(detected_errors) if detected_errors else "감지된 주요 오류 없음"
    cue_text = cue or "현재 리듬을 유지하면서 발바닥 전체 압력을 고르게 유지하세요."
    return "\n".join([
        "[전체 평가]",
        "SET 1에서는 기준 자세를 수집했고, SET 2에서는 live feedback을 보면서 같은 3회 반복을 수행했습니다.",
        "mock backend 기준으로 SET 2에서 rule issue 수와 자세 흔들림이 줄어드는 흐름을 확인할 수 있습니다.",
        "",
        "[가장 중요한 문제]",
        f"{issue_text}입니다. 감지된 오류 목록: {detected_text}.",
        "",
        "[다음 반복에서 신경 쓸 점]",
        f"1. {cue_text}",
        "2. 내려갈 때 속도를 늦추고, 올라올 때 무릎과 발끝 방향이 같이 움직이는지 확인하세요.",
        "",
        "[SET 1/SET 2 개선점]",
        "SET 1은 live feedback 없이 기준 데이터를 모으는 구간이고, SET 2는 cue를 보며 수정하는 구간입니다.",
        "QR report의 SET 비교 표에서 오류 수, cue 수, 상체 각도, 무릎 정렬 peak 변화를 확인하세요.",
    ])


def cleanup_session_files(paths: set[Path], output_root: Path | None = None) -> int:
    deleted = 0
    if output_root is None:
        output_root = Path(__file__).resolve().parent / "outputs"
    output_root = output_root.resolve()
    for path in sorted(paths, key=lambda p: len(str(p)), reverse=True):
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if not resolved.exists():
            continue
        if not resolved.is_file():
            continue
        try:
            resolved.relative_to(output_root)
        except ValueError:
            continue
        resolved.unlink()
        deleted += 1
    return deleted


if __name__ == "__main__":
    raise SystemExit(main())
