from __future__ import annotations


ISSUE_INFO = {
    "KneeValgus": {
        "label": "무릎 정렬",
        "summary": "무릎이 발끝 방향보다 안쪽으로 모이는 경향이 보입니다.",
        "priority": "다음 반복에서는 무릎이 발끝 방향을 따라가게 만드는 것을 가장 먼저 신경 쓰세요.",
        "cues": [
            "내려가고 올라올 때 양쪽 무릎을 두 번째 발가락 방향으로 밀어주세요.",
            "발바닥 전체로 바닥을 누르면서 무릎이 안쪽으로 무너지지 않는지 확인하세요.",
        ],
    },
    "ShallowDepth": {
        "label": "스쿼트 깊이",
        "summary": "최하단 깊이가 충분하지 않게 포착됐습니다.",
        "priority": "자세가 무너지지 않는 범위에서 엉덩이를 조금 더 낮추는 것이 우선입니다.",
        "cues": [
            "내려가는 속도를 조금 늦추고 최하단에서 잠깐 멈춰 깊이를 확인하세요.",
            "상체를 세운 채 엉덩이가 뒤로만 빠지지 않도록 발바닥 전체로 버티세요.",
        ],
    },
    "TorsoLean": {
        "label": "상체 기울기",
        "summary": "상체가 앞으로 많이 기울어진 경향이 보입니다.",
        "priority": "상체를 조금 더 세우고 골반과 가슴이 같이 움직이게 만드는 것이 중요합니다.",
        "cues": [
            "가슴을 정면에 보여준다는 느낌으로 내려가세요.",
            "올라올 때 엉덩이만 먼저 오르지 않게 가슴과 골반을 같이 올리세요.",
        ],
    },
    "WeightAsymmetry": {
        "label": "좌우 체중 균형",
        "summary": "좌우 체중이 한쪽으로 치우치는 경향이 보입니다.",
        "priority": "양발에 체중을 비슷하게 싣는 것을 먼저 맞추세요.",
        "cues": [
            "내려갈 때 양발 앞꿈치와 뒤꿈치가 모두 바닥을 고르게 누르는지 느껴보세요.",
            "올라올 때 몸통이 한쪽으로 빠지지 않는지 확인하세요.",
        ],
    },
    "CorePressureDrop": {
        "label": "복압 유지",
        "summary": "반복 중 복압 또는 브레이싱 유지가 약해지는 구간이 보입니다.",
        "priority": "내려가기 전 복부 압력을 만들고, 올라올 때까지 그 긴장을 유지하는 것이 우선입니다.",
        "cues": [
            "내려가기 전에 숨을 짧게 들이마시고 복부를 단단하게 잡아주세요.",
            "올라오는 중간에 힘이 풀리지 않도록 허리 벨트 주변 긴장을 유지하세요.",
        ],
    },
    "HeelLift": {
        "label": "뒤꿈치 들림",
        "summary": "뒤꿈치가 들리는 패턴이 감지되었습니다.",
        "priority": "발바닥 전체를 바닥에 붙이고 움직이는 것이 우선입니다.",
        "cues": [
            "내려갈 때 뒤꿈치가 뜨지 않도록 발바닥 전체로 바닥을 누르세요.",
            "필요하면 보폭을 조금 넓히고 발끝을 바깥쪽으로 열어보세요.",
        ],
    },
    "TorsoInstability": {
        "label": "몸통 안정성",
        "summary": "몸통이 흔들리는 패턴이 보입니다.",
        "priority": "속도보다 몸통을 안정적으로 고정하는 것을 먼저 신경 쓰세요.",
        "cues": [
            "내려가기 전에 복부에 힘을 주고 몸통을 단단하게 잡으세요.",
            "반동 없이 일정한 속도로 내려가고 올라오세요.",
        ],
    },
}

ISSUE_PRIORITY = [
    "TrackingUncertain",
    "KneeValgus",
    "HeelLift",
    "ShallowDepth",
    "TorsoLean",
    "WeightAsymmetry",
    "CorePressureDrop",
    "TorsoInstability",
]


class TemplateReporter:
    """Deterministic Korean report focused on actionable coaching."""

    REPORT_ALLOWED_ISSUES = {
        "TrackingUncertain",
        "CorePressureDrop",
        "WeightAsymmetry",
        "HeelLift",
        "TorsoLean",
        "TorsoInstability",
    }

    def generate(self, rep_summary: dict) -> str:
        quality = rep_summary.get("last_rep_quality", {}) or {}
        detected = self._detected_errors(rep_summary)
        main_issue = self._main_issue(rep_summary, detected)

        if quality.get("level") == "bad" or main_issue == "TrackingUncertain":
            return self._tracking_report(quality, rep_summary)

        if not detected:
            return self._clean_report(quality, rep_summary)

        info = ISSUE_INFO.get(main_issue, ISSUE_INFO.get(detected[0]))
        secondary = [ISSUE_INFO[code]["label"] for code in detected if code != main_issue and code in ISSUE_INFO]

        lines = [
            "[전체 평가]",
            info["summary"],
        ]
        if secondary:
            lines.append("함께 확인된 부분: " + ", ".join(secondary[:2]))

        lines.extend([
            "",
            "[가장 중요한 문제]",
            info["priority"],
            "",
            "[다음 반복에서 신경 쓸 점]",
        ])
        for index, cue in enumerate(info["cues"][:2], start=1):
            lines.append(f"{index}. {cue}")

        confidence = quality.get("mean_confidence")
        lower_vis = quality.get("mean_lower_body_visibility")
        if confidence is not None and lower_vis is not None and (confidence < 0.35 or lower_vis < 0.25):
            lines.extend([
                "",
                "촬영/추적 참고: 이번 분석은 카메라 추적 신뢰도가 낮은 편입니다. 다음 반복에서는 골반-무릎-발목-발끝이 화면에 잘 들어오게 맞추면 더 정확합니다.",
            ])

        lines.extend(["", "[SET 1/SET 2 개선점]"])
        lines.extend(self._set_comparison_lines(rep_summary))
        lines.extend(self._feedback_trace_lines(rep_summary))
        return "\n".join(lines)

    def _tracking_report(self, quality: dict, rep_summary: dict) -> str:
        lines = [
            "[전체 평가]",
            "자세 평가보다 카메라 추적 상태를 먼저 개선해야 합니다.",
            "",
            "[가장 중요한 문제]",
            "전신이 화면에 들어오지 않아 무릎, 깊이, 발 위치를 제대로 판단하기 어렵습니다.",
            "",
            "[다음 반복에서 신경 쓸 점]",
            "1. 카메라를 한 걸음 더 멀리 두고 골반, 무릎, 발목, 발끝이 모두 보이게 맞추세요.",
            "2. 첫 반복은 안정된 자세로 1초 정도 멈춘 뒤 스쿼트를 시작하세요.",
            "",
            "[SET 1/SET 2 개선점]",
        ]
        lines.extend(self._set_comparison_lines(rep_summary))
        lines.extend(self._feedback_trace_lines(rep_summary))
        return "\n".join(lines)

    def _clean_report(self, quality: dict, rep_summary: dict) -> str:
        confidence = quality.get("mean_confidence", 0.0) or 0.0
        if confidence < 0.35:
            return self._tracking_report(quality, rep_summary)

        lines = [
            "[전체 평가]",
            "큰 자세 오류는 뚜렷하게 감지되지 않았습니다.",
            "",
            "[가장 중요한 문제]",
            "새로운 교정보다는 같은 리듬과 안정적인 전신 추적을 유지하는 것이 좋습니다.",
            "",
            "[다음 반복에서 신경 쓸 점]",
            "1. 같은 속도로 천천히 내려가고 올라오세요.",
            "2. 무릎은 발끝 방향을 따라가고 발바닥 전체가 바닥에 붙어 있는지 유지하세요.",
            "",
            "[SET 1/SET 2 개선점]",
        ]
        lines.extend(self._set_comparison_lines(rep_summary))
        lines.extend(self._feedback_trace_lines(rep_summary))
        return "\n".join(lines)

    def _feedback_trace_lines(self, rep_summary: dict) -> list[str]:
        trace = rep_summary.get("feedback_trace") or []
        if not trace:
            return []
        lines = ["", "[Live Feedback Trace]"]
        for item in trace[:3]:
            phase = item.get("phase") or "-"
            issue = item.get("issue") or "None"
            cue = item.get("cue") or "-"
            reason = item.get("decision_reason") or "-"
            lines.append(f"- {phase}: {issue} → {cue} ({reason})")
        return lines

    def _set_comparison_lines(self, rep_summary: dict) -> list[str]:
        comparison = rep_summary.get("set_comparison") or {}
        if not isinstance(comparison, dict) or not comparison:
            return ["SET 1/SET 2 비교 정보가 아직 충분하지 않습니다. 두 세트가 모두 끝나면 자동으로 채워집니다."]

        lines: list[str] = []
        summary = comparison.get("summary")
        if summary:
            lines.append(str(summary))

        error_delta = comparison.get("error_count_delta")
        feedback_delta = comparison.get("feedback_count_delta")
        torso_delta = comparison.get("torso_angle_peak_delta")
        # Knee valgus is retained in raw comparison data for diagnostics only;
        # it is not part of the official report rubric.
        knee_delta = None

        if error_delta is not None:
            if error_delta < 0:
                lines.append(f"감지 오류 수가 SET 1보다 {abs(error_delta)}개 줄었습니다.")
            elif error_delta > 0:
                lines.append(f"감지 오류 수가 SET 1보다 {error_delta}개 늘어 추가 안정화가 필요합니다.")
            else:
                lines.append("감지 오류 수는 SET 1과 비슷했습니다.")

        if feedback_delta is not None:
            lines.append(f"Live feedback cue는 SET 1 대비 {feedback_delta:+}회 기록되었습니다.")

        if torso_delta is not None:
            lines.append(f"상체 기울기 peak 변화: {torso_delta:+.1f}도.")

        if knee_delta is not None:
            lines.append(f"무릎 정렬 peak 변화: {knee_delta:+.2f}.")

        return lines or ["SET 비교 payload는 있으나 요약 가능한 수치가 부족합니다."]

    def _main_issue(self, rep_summary: dict, detected: list[str]) -> str | None:
        rule = rep_summary.get("rule_analysis", {}) or {}
        main = rep_summary.get("main_issue") or rule.get("main_issue")
        if main in self.REPORT_ALLOWED_ISSUES:
            return main
        for code in ISSUE_PRIORITY:
            if code in detected:
                return code
        return detected[0] if detected else None

    def _detected_errors(self, rep_summary: dict) -> list[str]:
        rule = rep_summary.get("rule_analysis", {}) or {}
        raw = rep_summary.get("detected_errors") or rule.get("detected_errors") or []
        return [
            code
            for code in raw
            if code and code != "None" and code in self.REPORT_ALLOWED_ISSUES
        ]
