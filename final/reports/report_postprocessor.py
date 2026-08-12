from __future__ import annotations

import re


SECTION_TITLES = ("[전체 평가]", "[가장 중요한 문제]", "[다음 반복에서 신경 쓸 점]")

FORBIDDEN_REPLACEMENTS = {
    "무릎 골반 관계": "무릎 정렬",
    "부상 위험이 높습니다": "",
    "진단됩니다": "관찰됩니다",
    "의학적으로 문제가 있습니다": "",
}

ISSUE_KEYWORDS = {
    "KneeValgus": ("무릎 정렬", "무릎이 발끝", "무릎이 안쪽", "안쪽으로 모"),
    "ShallowDepth": ("깊이 부족", "최하단 깊이", "깊이가 부족", "엉덩이를 조금 더 낮"),
    "TorsoLean": ("상체가 과하게", "상체 기울", "앞으로 기울"),
    "WeightAsymmetry": ("좌우 체중", "체중 분배", "하중 균형", "압력 차이"),
    "HeelLift": ("뒤꿈치가 들", "발뒤꿈치가 들"),
    "TorsoInstability": ("몸통이 흔들", "몸통 안정"),
    "CorePressureDrop": ("복압", "코어", "브레이싱", "압력 유지"),
    "TrackingUncertain": ("카메라 추적", "자세 추적", "인식"),
}

# The final AI report may only discuss the official scoring rubric.
REPORT_ALLOWED_ISSUES = {
    "TrackingUncertain",
    "CorePressureDrop",
    "WeightAsymmetry",
    "HeelLift",
    "TorsoLean",
    "TorsoInstability",
}

NON_RUBRIC_TOPIC_RE = re.compile(
    r"(?i)(standing\s+pelvic\s+tuck|pelvic\s+tuck|knee\s+valgus|knee\s+alignment|"
    r"standing\s+pelvis|골반\s*말림|무릎\s*정렬|외반)"
)

MAIN_ISSUE_KEYWORDS = {
    "KneeValgus": ("무릎", "정렬", "발끝", "안쪽"),
    "ShallowDepth": ("깊이", "최하단", "낮"),
    "TorsoLean": ("상체", "몸통", "기울"),
    "WeightAsymmetry": ("좌우", "체중", "하중", "압력"),
    "HeelLift": ("뒤꿈치", "발뒤꿈치"),
    "TorsoInstability": ("몸통", "흔들", "안정"),
    "CorePressureDrop": ("복압", "코어", "브레이싱", "압력"),
    "TrackingUncertain": ("카메라", "추적", "인식"),
}


class ReportPostProcessor:
    """Keep AI coach reports aligned with rule-based rep_summary evidence."""

    def process(self, report: str, rep_summary: dict, fallback_text: str) -> str:
        if self._last_rep_quality_level(rep_summary) == "bad":
            return fallback_text
        if not report or self._looks_broken(report):
            return fallback_text

        cleaned = self._apply_forbidden_replacements(report.strip())
        cleaned = self._remove_unsupported_error_sentences(cleaned, rep_summary)
        cleaned = self._limit_cues(cleaned)
        cleaned = self._ensure_sections(cleaned, fallback_text)

        if cleaned == fallback_text:
            return fallback_text
        if not self._main_issue_present(cleaned, rep_summary):
            return fallback_text
        if self._looks_broken(cleaned):
            return fallback_text
        return cleaned.strip()

    def _last_rep_quality_level(self, rep_summary: dict) -> str | None:
        quality = rep_summary.get("last_rep_quality", {})
        if quality.get("level"):
            return quality.get("level")
        legacy_quality = rep_summary.get("tracking_quality", {})
        return legacy_quality.get("quality_level")

    def _apply_forbidden_replacements(self, text: str) -> str:
        cleaned = text
        for forbidden, replacement in FORBIDDEN_REPLACEMENTS.items():
            cleaned = cleaned.replace(forbidden, replacement)
        return cleaned

    def _remove_unsupported_error_sentences(self, text: str, rep_summary: dict) -> str:
        allowed = self._allowed_errors(rep_summary)
        unsupported = [
            code
            for code in ISSUE_KEYWORDS
            if code not in allowed or code not in REPORT_ALLOWED_ISSUES
        ]
        if not unsupported:
            return text

        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped in SECTION_TITLES:
                lines.append(line)
                continue
            if NON_RUBRIC_TOPIC_RE.search(stripped):
                continue
            if re.match(r"^\d+\.", stripped) or stripped.startswith("-"):
                if not self._mentions_any_issue(stripped, unsupported):
                    lines.append(line)
                continue
            sentences = self._split_sentences(line)
            kept = [sentence for sentence in sentences if not self._mentions_any_issue(sentence, unsupported)]
            if kept:
                lines.append(" ".join(kept))
        return "\n".join(lines)

    def _allowed_errors(self, rep_summary: dict) -> set[str]:
        rule = rep_summary.get("rule_analysis", {})
        detected = {
            code
            for code in (rep_summary.get("detected_errors") or rule.get("detected_errors") or [])
            if code in REPORT_ALLOWED_ISSUES
        }
        main_issue = rep_summary.get("main_issue") or rule.get("main_issue")
        if main_issue in REPORT_ALLOWED_ISSUES:
            detected.add(main_issue)
        return detected

    def _mentions_any_issue(self, text: str, issue_codes: list[str]) -> bool:
        for code in issue_codes:
            keywords = ISSUE_KEYWORDS.get(code, ())
            if any(keyword in text for keyword in keywords):
                return True
        return False

    def _split_sentences(self, text: str) -> list[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [part.strip() for part in parts if part.strip()]

    def _limit_cues(self, text: str) -> str:
        marker = "[다음 반복에서 신경 쓸 점]"
        if marker not in text:
            return text
        before, after = text.split(marker, 1)
        cue_block, suffix = self._split_at_next_section(after)
        cues = []
        for line in cue_block.strip().splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^\d+\.", stripped):
                cue = re.sub(r"^\d+\.\s*", "", stripped).strip()
                cues.append(cue)
            elif stripped.startswith("-"):
                cues.append(stripped.lstrip("-").strip())
        if not cues:
            return text
        numbered = [f"{index + 1}. {cue}" for index, cue in enumerate(cues[:2])]
        return before.rstrip() + "\n\n" + marker + "\n" + "\n".join(numbered) + suffix

    def _split_at_next_section(self, text: str) -> tuple[str, str]:
        next_section = re.search(r"\n(?=\[[^\]]+\])", text)
        if not next_section:
            return text, ""
        return text[: next_section.start()], text[next_section.start() :]

    def _ensure_sections(self, text: str, fallback_text: str) -> str:
        if all(section in text for section in SECTION_TITLES):
            return text
        return fallback_text

    def _main_issue_present(self, text: str, rep_summary: dict) -> bool:
        rule = rep_summary.get("rule_analysis", {})
        main_issue = rep_summary.get("main_issue") or rule.get("main_issue")
        if not main_issue or main_issue == "TrackingUncertain":
            return True
        keywords = MAIN_ISSUE_KEYWORDS.get(main_issue, ())
        return any(keyword in text for keyword in keywords)

    def _looks_broken(self, text: str) -> bool:
        if len(text.strip()) < 20:
            return True
        replacement_noise = text.count("�") + text.count("占") + text.count("??")
        return replacement_noise >= 3
