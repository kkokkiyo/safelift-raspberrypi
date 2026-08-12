# System Overview

`main.py` wires webcam input, MediaPipe pose tracking, rule-based analysis, phase captures, and post-rep reporting.

Runtime path:

```text
webcam -> MediaPipePoseTracker -> geometry metrics -> PhaseDetector
       -> RuleErrorDetector -> RiskAwareFeedbackEngine -> Dashboard
       -> PhaseCaptureManager -> RepSummaryBuilder -> GeminiReporter/TemplateReporter
```

Gemini is not used for real-time decisions. It is only called after a rep summary is ready, and `TemplateReporter` is used when Gemini is disabled or unavailable.
