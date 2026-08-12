import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.scoring import evaluate


class TestScoring(unittest.TestCase):
    def test_balanced_complete_frame_is_high_score(self):
        frame = {"left_pressure_ratio": 0.5, "right_pressure_ratio": 0.5, "core_pressure_ratio": 1.0, "imu_pitch": 10.0, "valid": True}
        pose = {"torso_angle": 10.0, "knee_angle": 90.0, "confidence": 0.9}
        result = evaluate(frame, pose, {"core_pressure_ratio": 1.0, "imu_pitch": 0.0})
        self.assertGreaterEqual(result["score"], 90)
        self.assertEqual(result["evidence"], "complete")


    def test_unbalanced_frame_loses_balance_points(self):
        frame = {"left_pressure_ratio": 0.8, "right_pressure_ratio": 0.2, "core_pressure_ratio": 0.8, "imu_pitch": 25.0, "valid": True}
        pose = {"torso_angle": 5.0, "knee_angle": 90.0, "confidence": 0.9}
        result = evaluate(frame, pose, {"core_pressure_ratio": 1.0, "imu_pitch": 0.0})
        self.assertLess(result["balance"], 23)
        self.assertLess(result["score"], 90)
