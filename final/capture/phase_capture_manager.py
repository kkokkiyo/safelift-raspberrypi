from pathlib import Path

import cv2

from analysis.frame_state import FrameState


class PhaseCaptureManager:
    def __init__(self, output_dir: str = "outputs/captures") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.candidates: dict[str, tuple[float, object]] = {}

    def reset(self) -> None:
        self.candidates.clear()

    def update(self, frame, frame_state: FrameState) -> None:
        phase = frame_state.phase
        if phase not in {"Standing", "Descent", "Bottom", "Ascent"}:
            return
        score = self._score(frame_state)
        current = self.candidates.get(phase)
        if current is None or score > current[0]:
            self.candidates[phase] = (score, frame.copy())

    def finalize_rep(self, rep_id: int) -> dict:
        snapshots = {}
        name_map = {
            "Standing": "standing",
            "Descent": "descent",
            "Bottom": "bottom",
            "Ascent": "ascent",
        }
        for phase, key in name_map.items():
            candidate = self.candidates.get(phase)
            if candidate is None:
                continue
            path = self.output_dir / f"rep{rep_id}_{key}.jpg"
            cv2.imwrite(str(path), candidate[1])
            snapshots[key] = str(path)
        return snapshots

    def _score(self, frame_state: FrameState) -> float:
        metrics = frame_state.metrics
        if frame_state.phase == "Standing":
            return float(frame_state.confidence)
        if frame_state.phase == "Descent":
            return float(metrics.get("torso_angle") or 0.0)
        if frame_state.phase == "Bottom":
            return float(metrics.get("hip_height") or 0.0)
        if frame_state.phase == "Ascent":
            return float(metrics.get("knee_valgus_score") or 0.0)
        return 0.0
