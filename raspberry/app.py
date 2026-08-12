from __future__ import annotations

import argparse
import threading
import time

from flask import Flask, Response, jsonify, render_template_string, request

from analysis.calibration import Calibration
from analysis.risk import assess
from analysis.scoring import evaluate
from hardware.direct_sensor_provider import DirectSensorProvider
from vision.pose import PoseAnalyzer


HTML = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SafeLift Raspberry Pi</title>
<style>
body{font-family:Arial,'Malgun Gothic',sans-serif;background:#101827;color:#e5edf5;margin:0}header{padding:18px 24px;background:#16243a;font-size:22px;font-weight:bold}main{display:grid;grid-template-columns:minmax(420px,1.3fr) minmax(340px,.9fr);gap:16px;padding:16px}.card{background:#172235;border:1px solid #2b3b54;border-radius:10px;padding:16px;margin-bottom:16px}img{width:100%;background:#05090f;border-radius:8px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.metric{background:#0f1928;border-radius:8px;padding:12px}.label{color:#9fb0c5;font-size:12px}.value{font-size:23px;margin-top:5px}.normal{color:#72e6a3}.warning{color:#ffd166}.danger{color:#ff7d7d}button{border:0;border-radius:6px;padding:10px 12px;background:#2779e6;color:white;font-weight:bold;margin-right:7px;cursor:pointer}pre{white-space:pre-wrap;line-height:1.5;color:#c9d5e3}@media(max-width:850px){main{grid-template-columns:1fr}}
</style></head><body><header>SafeLift · Raspberry Pi 멀티센서 자세 분석</header><main>
<section><div class="card"><h2>Camera / MediaPipe</h2><img src="/camera.mjpg" alt="camera"></div><div class="card"><h2>센서 원시 상태</h2><pre id="sensor">-</pre></div></section>
<section><div class="card"><button onclick="post('/api/calibration/start')">캘리브레이션 시작</button><button onclick="post('/api/session/start')">측정 시작</button><button onclick="post('/api/session/stop')">측정 종료</button><p id="status">대기 중</p></div><div class="card"><h2>결과</h2><div class="grid"><div class="metric"><div class="label">종합 점수</div><div id="score" class="value">-</div></div><div class="metric"><div class="label">위험도</div><div id="risk" class="value">-</div></div><div class="metric"><div class="label">운동 단계</div><div id="phase" class="value">-</div></div><div class="metric"><div class="label">신뢰도</div><div id="confidence" class="value">-</div></div></div></div><div class="card"><h2>판정 근거</h2><pre id="details">-</pre></div></section></main>
<script>
async function post(url){const r=await fetch(url,{method:'POST'});const d=await r.json();document.getElementById('status').textContent=d.message||'완료';}
async function tick(){try{const s=await (await fetch('/api/state',{cache:'no-store'})).json();const score=s.score||{};const risk=s.risk||{};document.getElementById('score').textContent=(score.score??'-')+' / 100';document.getElementById('risk').textContent=risk.level||'-';document.getElementById('risk').className='value '+(risk.level||'');document.getElementById('phase').textContent=s.phase||'-';document.getElementById('confidence').textContent=((s.pose||{}).confidence??0).toFixed(2);document.getElementById('sensor').textContent=JSON.stringify(s.sensor||{},null,2);document.getElementById('details').textContent=JSON.stringify({score:score,risk:risk,pose:s.pose||{},calibration:s.calibration||{}},null,2);}catch(e){document.getElementById('status').textContent='서버 연결 대기';}setTimeout(tick,500)}tick();
</script></body></html>"""


class SafeLiftApp:
    def __init__(self, mock: bool, camera_index: int | None, no_camera: bool):
        self.mock = mock
        self.no_camera = no_camera
        self.sensor = DirectSensorProvider(mock=mock)
        self.pose = PoseAnalyzer(enabled=not no_camera)
        self.calibration = Calibration()
        self.running = True
        self.session = False
        self.lock = threading.Lock()
        self.latest = {"sensor": {}, "pose": {}, "score": {}, "risk": {}, "phase": "standing", "calibration": {}}
        self.camera = None
        self.camera_error = None
        if not no_camera:
            try:
                import cv2
                self.camera = cv2.VideoCapture(0 if camera_index is None else camera_index)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if not self.camera.isOpened():
                    self.camera_error = "camera could not be opened"
            except Exception as exc:
                self.camera_error = str(exc)

    def loop(self):
        while self.running:
            try:
                frame = self.sensor.read()
                pose_data, image = (None, None)
                if self.camera is not None and self.camera.isOpened():
                    ok, image = self.camera.read()
                    if ok:
                        pose_data, image = self.pose.analyze(image)
                pose_data = pose_data or {"confidence": 0.0}
                cal = self.calibration.update(frame)
                sensor_dict = frame.to_dict()
                score = evaluate(sensor_dict, pose_data, cal.get("baseline") or {})
                risk = assess(sensor_dict, pose_data, score)
                with self.lock:
                    self.latest = {"timestamp": time.time(), "sensor": sensor_dict, "pose": pose_data, "score": score, "risk": risk, "phase": self._phase(pose_data), "calibration": cal, "session": self.session}
                    if image is not None:
                        import cv2
                        ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                        if ok: self.latest["jpeg"] = encoded.tobytes()
            except Exception as exc:
                with self.lock:
                    self.latest["runtime_error"] = str(exc)
            time.sleep(0.02)

    @staticmethod
    def _phase(pose: dict) -> str:
        knee = pose.get("knee_angle")
        if knee is None: return "unknown"
        if knee > 155: return "standing"
        if knee > 100: return "descent"
        if knee > 70: return "bottom"
        return "ascent"

    def state(self):
        with self.lock:
            return {k: v for k, v in self.latest.items() if k != "jpeg"}

    def jpeg(self):
        with self.lock:
            return self.latest.get("jpeg")

    def close(self):
        self.running = False
        if self.camera is not None: self.camera.release()
        self.pose.close()
        self.sensor.close()


def create_app(worker: SafeLiftApp) -> Flask:
    app = Flask(__name__)

    @app.get('/')
    def index(): return render_template_string(HTML)

    @app.get('/api/state')
    def state(): return jsonify(worker.state())

    @app.get('/api/health')
    def health():
        state = worker.state()
        return jsonify({"ok": "runtime_error" not in state, "camera": worker.camera is not None and worker.camera.isOpened(), "camera_error": worker.camera_error, "mock": worker.mock, "sensor": state.get("sensor", {}), "runtime_error": state.get("runtime_error")})

    @app.post('/api/calibration/start')
    def calibration_start(): worker.calibration.reset(); return jsonify({"ok": True, "message": "캘리브레이션을 시작했습니다."})

    @app.post('/api/session/start')
    def session_start(): worker.session = True; return jsonify({"ok": True, "message": "측정을 시작했습니다."})

    @app.post('/api/session/stop')
    def session_stop(): worker.session = False; return jsonify({"ok": True, "message": "측정을 종료했습니다."})

    @app.get('/camera.mjpg')
    def camera_stream():
        def stream():
            while worker.running:
                image = worker.jpeg()
                if image:
                    yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + image + b'\r\n'
                time.sleep(0.05)
        return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true', help='Use fake FSR/IMU/pressure values')
    parser.add_argument('--no-camera', action='store_true')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8090)
    args = parser.parse_args()
    worker = SafeLiftApp(args.mock, args.camera, args.no_camera)
    thread = threading.Thread(target=worker.loop, daemon=True)
    thread.start()
    try:
        create_app(worker).run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        worker.close()


if __name__ == '__main__':
    main()
