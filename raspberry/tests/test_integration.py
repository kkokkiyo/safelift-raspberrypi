import sys
import time
import importlib.util
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FLASK_AVAILABLE = importlib.util.find_spec('flask') is not None

if FLASK_AVAILABLE:
    from app import SafeLiftApp, create_app


@unittest.skipUnless(FLASK_AVAILABLE, 'Flask is not installed; install requirements.txt')
class TestIntegration(unittest.TestCase):
    def test_mock_pipeline_updates_state_and_health(self):
        worker = SafeLiftApp(mock=True, camera_index=0, no_camera=True)
        try:
            import threading
            thread = threading.Thread(target=worker.loop, daemon=True)
            thread.start()
            time.sleep(0.15)
            client = create_app(worker).test_client()
            state = client.get('/api/state')
            health = client.get('/api/health')
            self.assertEqual(state.status_code, 200)
            payload = state.get_json()
            self.assertTrue(payload['sensor']['valid'])
            self.assertIn('score', payload)
            self.assertIn('risk', payload)
            self.assertEqual(health.status_code, 200)
            self.assertTrue(health.get_json()['mock'])
        finally:
            worker.close()

    def test_control_endpoints(self):
        worker = SafeLiftApp(mock=True, camera_index=0, no_camera=True)
        try:
            client = create_app(worker).test_client()
            self.assertTrue(client.post('/api/session/start').get_json()['ok'])
            self.assertTrue(worker.session)
            self.assertTrue(client.post('/api/session/stop').get_json()['ok'])
            self.assertFalse(worker.session)
            self.assertTrue(client.post('/api/calibration/start').get_json()['ok'])
        finally:
            worker.close()
