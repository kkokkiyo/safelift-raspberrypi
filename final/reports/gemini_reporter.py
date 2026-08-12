"""Gemini-backed SafeLift coach report generation.

Compatible with the existing reporter interface: ``generate(rep_summary) -> str``.
The backend still falls back to the deterministic template reporter whenever
Gemini is not configured or the API call fails.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .report_postprocessor import ReportPostProcessor


class GeminiReporter:
    # Keep AI coaching aligned with the official SafeLift score rubric.
    # These are the only issue categories that may appear in the report.
    REPORT_ALLOWED_ISSUES = {
        "TrackingUncertain",
        "CorePressureDrop",
        "WeightAsymmetry",
        "HeelLift",
        "TorsoLean",
        "TorsoInstability",
    }

    def __init__(self, fallback: Any, model: str | None = None, api_key: str | None = None):
        self.fallback = fallback
        self.model = model if model is not None else os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        self.api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        self._last_status = "Gemini is not configured; template fallback is active."
        self.postprocessor = ReportPostProcessor()

    def status(self) -> str:
        return self._last_status

    def generate(self, rep_summary: dict) -> str:
        fallback_text = self.fallback.generate(rep_summary)
        if not self.api_key:
            self._last_status = "GEMINI_API_KEY is missing; template fallback used."
            return fallback_text

        try:
            from google import genai

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=self.model,
                contents=self._prompt(rep_summary),
            )
            text = (getattr(response, "text", None) or "").strip()
            if not text:
                raise RuntimeError("Gemini returned an empty response")
            processed = self.postprocessor.process(text, rep_summary, fallback_text)
            if processed == fallback_text:
                self._last_status = f"Gemini report rejected by evidence guard; template fallback used."
                return fallback_text
            self._last_status = f"Gemini report generated with {self.model} and evidence guard."
            return processed
        except Exception as exc:
            self._last_status = f"Gemini unavailable ({exc}); template fallback used."
            return fallback_text

    @staticmethod
    def build_evidence_payload(rep_summary: dict) -> dict:
        """Return the exact evidence shape Gemini is allowed to use.

        The payload is deliberately explicit so verification can prove the
        report pipeline is grounded in the manual-required evidence:
        phase captures, phase quality, sensor summary, detected errors, live
        feedback cues, feedback decision trace, and SET 1 vs SET 2 comparison. Raw camera frames are not
        sent here; only backend-created paths and summaries are included.
        """

        rule = rep_summary.get("rule_analysis") or {}
        quality = rep_summary.get("last_rep_quality") or {}
        detected_errors = [
            code
            for code in (rep_summary.get("detected_errors") or rule.get("detected_errors") or [])
            if code in GeminiReporter.REPORT_ALLOWED_ISSUES
        ]
        main_issue = rep_summary.get("main_issue") or rule.get("main_issue")
        if main_issue not in GeminiReporter.REPORT_ALLOWED_ISSUES:
            main_issue = None
        metrics = rep_summary.get("metrics") or {}
        rubric_metrics = {
            key: metrics[key]
            for key in (
                "core_pressure_min_ratio",
                "fsr_scoring_type",
                "forefoot_shift_peak_percent",
                "heel_pressure_drop",
                "max_side_torso_angle",
                "max_torso_angle",
                "imu_pitch_peak",
            )
            if key in metrics
        }
        comparison = rep_summary.get("set_comparison") or {}
        rubric_comparison = {
            key: comparison[key]
            for key in (
                "summary",
                "error_count_delta",
                "feedback_count_delta",
                "torso_angle_peak_delta",
            )
            if key in comparison
        }
        sensor_summary = rep_summary.get("sensor_summary") or {}
        rubric_sensor_summary = {
            key: sensor_summary[key]
            for key in (
                "core_pressure_ratio",
                "left_pressure_ratio",
                "right_pressure_ratio",
                "heel_pressure_drop_count",
                "forefoot_shift_percent",
                "fsr_scoring_type",
                "imu_pitch",
            )
            if key in sensor_summary
        }
        feedback_trace = [
            item
            for item in (rep_summary.get("feedback_trace") or [])
            if isinstance(item, dict)
            and item.get("issue") in GeminiReporter.REPORT_ALLOWED_ISSUES
        ]
        feedback_history = [
            item
            for item in (
                rep_summary.get("selected_feedback_history")
                or rule.get("selected_realtime_feedback")
                or []
            )
            if not isinstance(item, dict)
            or item.get("issue") in GeminiReporter.REPORT_ALLOWED_ISSUES
        ]
        selected_feedback = rep_summary.get("selected_feedback")
        if isinstance(selected_feedback, dict) and (
            selected_feedback.get("issue") not in GeminiReporter.REPORT_ALLOWED_ISSUES
        ):
            selected_feedback = None
        return {
            "rep_id": rep_summary.get("rep_id"),
            "report_source": rep_summary.get("report_source"),
            "sensor_mode": rep_summary.get("sensor_mode"),
            "phase_snapshots": rep_summary.get("phase_snapshots") or {},
            "phase_quality": rep_summary.get("phase_quality") or quality.get("phase_quality") or {},
            "metrics": rubric_metrics,
            "sensor_summary": rubric_sensor_summary,
            "detected_errors": detected_errors,
            "main_issue": main_issue,
            "selected_feedback": selected_feedback,
            "selected_feedback_history": feedback_history,
            "feedback_trace": feedback_trace,
            "set_comparison": rubric_comparison,
            "score_summary": rep_summary.get("score_summary") or {},
            "score_analysis": rep_summary.get("score_analysis") or {},
            "tracking_quality": quality,
        }

    @classmethod
    def _prompt(cls, rep_summary: dict) -> str:
        payload = json.dumps(cls.build_evidence_payload(rep_summary), ensure_ascii=False, indent=2, default=str)
        return f"""You are SafeLift, a cautious squat-form coach. Analyze only the evidence below.
Do not diagnose medical conditions. Never invent an error absent from detected_errors.
Use only the official SafeLift scoring rubric: Core Bracing (35), FSR Ground Balance (35),
and Spine Alignment (30). You may also mention tracking quality, sensor confidence,
and SET 1 versus SET 2 changes that are directly present in the evidence.
The score_analysis object contains the six individual REP scores. Compare
set1_best and set2_best when discussing the feedback effect; do not average
the six REP scores. If set2_best is higher, calculate the improvement rate as
((set2_best - set1_best) / set1_best) * 100 and round it to one decimal place.
Use the exact sentence format: "실시간 피드백을 제공 받은 Set2가 Set1에 비해
X.X%의 점수 향상률을 보였습니다." If set1_best is higher, write exactly:
"다음 번엔 실시간 피드백에 집중해서 스쿼트를 해주세요!"
For coaching, use score_analysis.lowest_rep and score_analysis.highest_rep
explicitly. The lowest REP is the primary evidence for what needs improvement;
the highest REP is the primary evidence for what the user did well. Inspect the
official-rubric deductions inside those REP records and prioritize the largest
deduction on the lowest REP. Do not choose a generic issue from an aggregate
summary when the lowest REP contains a larger official-rubric deduction.
Mention the highest REP's strongest area as a separate positive point. If the
lowest REP and highest REP are the same record, state that the available REP
evidence is limited and do not invent a comparison.
Do not discuss, score, penalize, or recommend corrections for any non-rubric posture topic.
In particular, never mention standing pelvic tuck, pelvic tuck, knee valgus, knee alignment,
or any other posture metric not included in the official rubric.
Return a concise Korean coaching report. Use these exact section titles:
[전체 평가]
[가장 중요한 문제]
[다음 반복에서 신경 쓸 점]
[SET 1/SET 2 개선점]
In the cue section, include maximum 2 numbered cues.
Mention when tracking or sensor quality limits confidence. Use only the evidence JSON.

Evidence JSON:
{payload}
"""
