from __future__ import annotations


PHASE_LABELS = {
    "standing": "Standing",
    "descent": "Descent",
    "bottom": "Bottom",
    "ascent": "Ascent",
}


def build_quality_evidence_text(rep_summary: dict) -> str:
    quality = rep_summary.get("last_rep_quality") or {}
    phase_quality = rep_summary.get("phase_quality") or quality.get("phase_quality") or {}
    rule = rep_summary.get("rule_analysis") or {}
    metrics = rep_summary.get("metrics") or {}

    lines = [
        "[수집 자료 및 판단 근거]",
        f"- Report Source: {rep_summary.get('report_source', 'phase_captures + no_sensor_summary')}",
        f"- Sensor Mode: {rep_summary.get('sensor_mode', '-')}",
        f"- Last Rep Quality: {str(quality.get('level', '-')).upper()}",
        f"- Report Allowed: {'YES' if quality.get('report_allowed') else 'NO'}",
        f"- Mean Confidence: {_fmt_float(quality.get('mean_confidence'))}",
        f"- Mean Lower Body Visibility: {_fmt_float(quality.get('mean_lower_body_visibility'))}",
        f"- Quality Reason: {quality.get('reason', '-')}",
        f"- Rule Main Issue: {rep_summary.get('main_issue') or rule.get('main_issue') or 'None'}",
        f"- Rule Detected Errors: {', '.join(rep_summary.get('detected_errors') or rule.get('detected_errors') or []) or 'None'}",
        f"- Selected Feedback: {rep_summary.get('selected_feedback') or 'None'}",
        f"- Key Metrics: min_knee_angle={_fmt_float(metrics.get('min_knee_angle'))}, "
        f"max_torso_angle={_fmt_float(metrics.get('max_torso_angle'))}, "
        f"max_knee_valgus_score={_fmt_float(metrics.get('max_knee_valgus_score'))}",
        "",
        "[Phase Evidence]",
    ]

    for phase in ("standing", "descent", "bottom", "ascent"):
        item = phase_quality.get(phase) or {}
        confidence = item.get("confidence", item.get("mean_confidence"))
        lower_visibility = item.get("lower_body_visibility", item.get("mean_lower_body_visibility"))
        quality_level = item.get("quality", item.get("level", "-"))
        lines.append(
            "- "
            f"{PHASE_LABELS[phase]}: {str(quality_level).upper()} | "
            f"frames={item.get('frame_count', 0)} | "
            f"capture={'YES' if item.get('has_capture') else 'NO'} | "
            f"confidence={_fmt_float(confidence)} | "
            f"lower_visibility={_fmt_float(lower_visibility)} | "
            f"reason={item.get('reason', '-')}"
        )

    return "\n".join(lines)


def _fmt_float(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)
