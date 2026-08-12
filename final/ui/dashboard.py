import json
import io
import math
import os
import re
import secrets
import html
import threading
import time
from pathlib import Path
from urllib.parse import quote, unquote
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import cv2
import numpy as np

from analysis.live_feedback_contract import describe_live_feedback


WHITE = (245, 245, 245)
GREEN = (60, 210, 90)
YELLOW = (40, 220, 220)
RED = (60, 60, 230)
CYAN = (230, 210, 80)


def _unity_exception_payload(
    calibration_status: str,
    live_quality: str,
    sensor_ready: bool,
    sensor_status: str,
    report_timeout: bool = False,
    report_status: str = "none",
) -> dict:
    """Return the manual-required exception state for Unity.

    Backend-disconnected is handled inside Unity because no backend response can
    carry that state. The remaining exceptions are published in every
    /api/session_state response so the HMD can show one clear operational
    notice instead of silently waiting.
    """

    calibration = str(calibration_status or "").strip().lower()
    quality = str(live_quality or "").strip().upper().replace(" ", "_")
    sensor_text = str(sensor_status or "").strip()
    report = str(report_status or "none").strip().lower()

    if report_timeout or report == "timeout":
        return {
            "code": "ReportTimeout",
            "level": "warning",
            "message": "AI 리포트 생성이 지연되고 있습니다. 기본 분석 결과를 먼저 확인해주세요.",
        }
    if any(token in calibration for token in ("fail", "error", "retry")):
        return {
            "code": "CalibrationFailed",
            "level": "error",
            "message": "초기 보정에 실패했습니다. 다시 측정해주세요.",
        }
    if not sensor_ready:
        suffix = f" ({sensor_text})" if sensor_text else ""
        return {
            "code": "SensorDisconnected",
            "level": "warning",
            "message": "센서 연결이 불안정합니다. Mock sensor 값으로 표시합니다." + suffix,
        }
    if quality in {"BAD", "LOW", "POOR", "UNCERTAIN", "TRACKING_BAD", "TRACKINGUNCERTAIN"}:
        return {
            "code": "TrackingBad",
            "level": "warning",
            "message": "전신이 화면에 들어오도록 위치를 조정해주세요.",
        }
    return {"code": "None", "level": "ok", "message": ""}


class DashboardRenderer:
    def __init__(self, width: int = 1500, height: int = 1020) -> None:
        self.width = width
        self.height = height

    def render(self, webcam_frame, frame_state, fps: float, report_text: str, phase_images: dict | None = None):
        canvas = np.full((self.height, self.width, 3), (24, 24, 24), dtype=np.uint8)
        # The selected OBS half is a native 640x480 camera image. Keep the
        # live preview at that exact 4:3 size instead of stretching it into
        # the old 50% x 61% dashboard area.
        left_w = 640
        top_h = 480
        live = self._fit(webcam_frame, left_w, top_h)
        canvas[0 : live.shape[0], 0 : live.shape[1]] = live
        cv2.rectangle(canvas, (0, 0), (left_w - 1, top_h - 1), (80, 80, 80), 1)

        x = left_w + 22
        y = 36
        self._put(canvas, "SafeLift XR", x, y, 0.8, CYAN, 2)
        y += 42
        self._put(canvas, f"FPS: {fps:.1f}", x, y)
        y += 30
        self._put(canvas, f"Phase: {frame_state.phase}", x, y, color=GREEN if frame_state.phase != "Unknown" else YELLOW)
        y += 30
        self._put(canvas, f"Confidence: {frame_state.confidence:.2f}", x, y)
        y += 30
        self._put(canvas, f"Feedback: {frame_state.realtime_feedback or '-'}", x, y, color=YELLOW)
        y += 40
        self._put(canvas, "Metrics", x, y, 0.65, CYAN, 2)
        for key in ["avg_knee_angle", "torso_angle", "hip_height", "hip_velocity", "knee_valgus_score"]:
            y += 26
            self._put(canvas, f"{key}: {self._fmt(frame_state.metrics.get(key))}", x, y, 0.55)
        y += 38
        self._put(canvas, "Mock/Serial Sensor", x, y, 0.65, CYAN, 2)
        sensor_lines = [
            f"source: {frame_state.sensor.source}",
            f"L/R pressure: {self._fmt(frame_state.sensor.left_pressure_ratio)} / {self._fmt(frame_state.sensor.right_pressure_ratio)}",
            f"heel drop: {frame_state.sensor.heel_pressure_drop}",
            f"imu pitch/roll: {self._fmt(frame_state.sensor.imu_pitch)} / {self._fmt(frame_state.sensor.imu_roll)}",
        ]
        for line in sensor_lines:
            y += 26
            self._put(canvas, line, x, y, 0.55)
        y += 36
        errors = ", ".join(frame_state.detected_errors) if frame_state.detected_errors else "None"
        self._put(canvas, f"Errors: {errors}", x, y, color=RED if frame_state.detected_errors else GREEN)
        y += 30
        self._put(canvas, f"Reason: {frame_state.decision_reason}", x, y, 0.52, WHITE)

        report_y = top_h + 34
        self._put(canvas, "AI Coach Report", 20, report_y, 0.7, CYAN, 2)
        display_report = "Korean report is shown in the browser panel. Press Space/Enter after a rep." if report_text else "Press Space/Enter after a rep to generate a report."
        for i, line in enumerate(self._wrap(display_report, 92)[:7]):
            self._put(canvas, line, 20, report_y + 32 + i * 24, 0.55, WHITE)

        if phase_images:
            thumb_x = left_w + 20
            thumb_y = top_h + 24
            self._put(canvas, "Phase Captures", thumb_x, thumb_y, 0.65, CYAN, 2)
            for i, (phase, img) in enumerate(phase_images.items()):
                if img is None:
                    continue
                thumb = self._fit(img, 250, 140)
                px = thumb_x + (i % 2) * 300
                py = thumb_y + 30 + (i // 2) * 180
                self._paste(canvas, thumb, px, py)
                label_y = min(py + thumb.shape[0] + 18, self.height - 8)
                if 0 <= label_y < self.height:
                    self._put(canvas, phase, px, label_y, 0.45, WHITE)

        return canvas

    def _fit(self, image, max_w: int, max_h: int):
        h, w = image.shape[:2]
        scale = min(max_w / w, max_h / h)
        return cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    def _put(self, canvas, text: str, x: int, y: int, scale: float = 0.58, color=WHITE, thickness: int = 1) -> None:
        cv2.putText(canvas, str(text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def _paste(self, canvas, image, x: int, y: int) -> None:
        if x >= self.width or y >= self.height:
            return
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(self.width, x + image.shape[1])
        y1 = min(self.height, y + image.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        src_x0 = x0 - x
        src_y0 = y0 - y
        canvas[y0:y1, x0:x1] = image[src_y0 : src_y0 + (y1 - y0), src_x0 : src_x0 + (x1 - x0)]

    def _fmt(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.3f}" if abs(value) < 1 else f"{value:.1f}"
        return str(value)

    def _wrap(self, text: str, max_len: int) -> list[str]:
        lines: list[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            while len(line) > max_len:
                lines.append(line[:max_len])
                line = line[max_len:]
            lines.append(line)
        return lines


class BrowserDashboard:
    def __init__(self, host: str = "127.0.0.1", port: int = 8090, max_width: int = 1280, sensor_push_provider: Any | None = None, static_report_root: str | None = None, public_report_base: str = "http://squartreport.ddnsfree.com/reports") -> None:
        self.host = host
        self.port = port
        self.max_width = max_width
        self.condition = threading.Condition()
        self.command_lock = threading.Lock()
        self.unity_score_lock = threading.Lock()
        self.pending_commands: list[str] = []
        self.unity_rep_scores: dict[str, dict] = {}
        self.latest_jpeg: bytes | None = None
        self.latest_live_jpeg: bytes | None = None
        self.latest_phase_jpegs: dict[str, bytes] = {}
        self.latest_state: bytes = b"{}"
        self.latest_report_id: str = ""
        self.archived_reports: dict[str, dict] = {}
        self.archived_phase_jpegs: dict[str, dict[str, bytes]] = {}
        self.report_archive_dir = Path(__file__).resolve().parents[1] / "report_archive"
        self.static_report_root = Path(static_report_root).expanduser() if static_report_root else None
        self.public_report_base = public_report_base.rstrip("/")
        self.static_report_tokens: dict[str, str] = {}
        self.latest_static_token: str = ""
        self.running = False
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.sensor_push_provider = sensor_push_provider

    def start(self) -> None:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    html = _browser_html().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return
                if self.path == "/qr":
                    html_payload = parent.get_qr_page(self.headers.get("Host")).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(html_payload)))
                    self.end_headers()
                    self.wfile.write(html_payload)
                    return
                if self.path == "/health":
                    html_payload = parent.get_health_page(self.headers.get("Host")).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(html_payload)))
                    self.end_headers()
                    self.wfile.write(html_payload)
                    return
                if self.path in ("/report/latest", "/report/demo"):
                    html_payload = parent.get_report_page().encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(html_payload)))
                    self.end_headers()
                    self.wfile.write(html_payload)
                    return
                if self.path.startswith("/report/"):
                    report_id = unquote(self.path.split("/", 2)[-1]).strip("/")
                    html_payload = parent.get_report_page(report_id=report_id).encode("utf-8")
                    status_code = 200 if parent.has_archived_report(report_id) else 404
                    self.send_response(status_code)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(html_payload)))
                    self.end_headers()
                    self.wfile.write(html_payload)
                    return
                if self.path == "/state.json":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(parent.latest_state)
                    return
                # Use the manual's /api/session_state endpoint as the Unity
                # client path. Keep /api/state as a compatibility alias for
                # older copied scripts.
                if self.path in ("/api/state", "/api/session_state"):
                    response = json.dumps(parent.get_unity_state(), ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                if self.path == "/api/latest_report":
                    response = json.dumps(parent.get_unity_report(), ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                if self.path == "/api/health":
                    response = json.dumps(parent.get_health(self.headers.get("Host")), ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                if self.path == "/api/qr_link":
                    base_url = parent.get_share_base_url(self.headers.get("Host"))
                    report_path = parent.get_share_report_path()
                    report_url = f"{base_url}{report_path}"
                    response = json.dumps({
                        "url": report_url,
                        "latest_url": f"{base_url}/report/latest",
                        "legacy_url": f"{base_url}/report/demo",
                        "qr_url": f"{base_url}/qr",
                        "qr_image_url": _goqr_image_url(report_url),
                        "qr_provider": "goqr.me/api.qrserver.com",
                        "report_id": parent.latest_report_id,
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                if self.path == "/api/qr_code.png":
                    base_url = parent.get_share_base_url(self.headers.get("Host"))
                    report_url = f"{base_url}{parent.get_share_report_path()}"
                    self.send_response(302)
                    self.send_header("Location", _goqr_image_url(report_url))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    return
                if self.path == "/api/live_frame.jpg":
                    jpeg = parent.get_latest_live(0.1)
                    if jpeg is None:
                        self.send_error(404)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return
                if self.path.startswith("/api/phase_capture/") and self.path.endswith(".jpg"):
                    phase = self.path.rsplit("/", 1)[-1][:-4]
                    jpeg = parent.latest_phase_jpegs.get(phase)
                    if jpeg is None:
                        jpeg = _placeholder_phase_jpeg(phase)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return
                if self.path.startswith("/api/report_phase_capture/") and self.path.endswith(".jpg"):
                    parts = self.path.split("/")
                    report_id = unquote(parts[3]) if len(parts) >= 5 else ""
                    phase = parts[-1][:-4]
                    jpeg = parent.get_archived_phase_jpeg(report_id, phase)
                    if jpeg is None:
                        jpeg = _placeholder_phase_jpeg(phase)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg)
                    return
                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    while parent.running:
                        jpeg = parent.get_latest(1.0)
                        if jpeg is None:
                            continue
                        try:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(jpeg + b"\r\n")
                        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                            break
                    return
                self.send_error(404)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    payload = {}
                if self.path == "/api/sensor_frame":
                    if parent.sensor_push_provider is None or not hasattr(parent.sensor_push_provider, "push"):
                        response = json.dumps({"ok": False, "error": "HTTP sensor provider is not enabled"}, ensure_ascii=False).encode("utf-8")
                        self.send_response(409)
                    elif not isinstance(payload, dict):
                        response = json.dumps({"ok": False, "error": "sensor frame must be a JSON object"}, ensure_ascii=False).encode("utf-8")
                        self.send_response(400)
                    else:
                        frame = parent.sensor_push_provider.push(payload)
                        response = json.dumps({
                            "ok": bool(frame.contract_ok),
                            "source": frame.source,
                            "contract_ok": frame.contract_ok,
                            "contract_errors": list(frame.contract_errors),
                        }, ensure_ascii=False).encode("utf-8")
                        self.send_response(200 if frame.contract_ok else 400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                if self.path == "/api/unity_rep_score":
                    if not isinstance(payload, dict):
                        response = b'{"ok":false,"error":"Unity REP score must be a JSON object"}'
                        self.send_response(400)
                    else:
                        accepted = parent.push_unity_rep_score(payload)
                        response = json.dumps({"ok": accepted}, ensure_ascii=False).encode("utf-8")
                        self.send_response(200 if accepted else 400)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(response)))
                    self.end_headers()
                    self.wfile.write(response)
                    return
                if self.path != "/command":
                    self.send_error(404)
                    return
                command = payload.get("command") if isinstance(payload, dict) else None
                if command in {"finish_rep", "reset_rep", "save_snapshot"}:
                    parent.push_command(command)
                    response = b'{"ok":true}'
                    self.send_response(200)
                else:
                    response = b'{"ok":false}'
                    self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.running = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        print(f"Browser dashboard: http://{self.host}:{self.port}")

    def update(self, frame, state: dict | None = None, phase_images: dict | None = None) -> None:
        if frame.shape[1] > self.max_width:
            scale = self.max_width / frame.shape[1]
            frame = cv2.resize(frame, (self.max_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return
        with self.condition:
            self.latest_jpeg = encoded.tobytes()
            if phase_images:
                for phase, image in phase_images.items():
                    if image is None:
                        continue
                    phase_ok, phase_encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                    if phase_ok:
                        self.latest_phase_jpegs[phase] = phase_encoded.tobytes()
            if state is not None:
                safe_state = _json_safe(state)
                self.latest_state = json.dumps(safe_state, ensure_ascii=False, allow_nan=False).encode("utf-8")
                self._archive_report_if_ready(safe_state)
            self.condition.notify_all()

    def update_live_frame(self, frame) -> None:
        if frame is None:
            return
        # Unity's RawImage receives this endpoint directly. Normalize the
        # live camera payload so a driver-returned composite/letterboxed frame
        # cannot carry its black padding into the Unity preview.
        if frame.shape[1] != 640 or frame.shape[0] != 480:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return
        with self.condition:
            self.latest_live_jpeg = encoded.tobytes()
            self.condition.notify_all()

    def update_phase_images(self, phase_images: dict | None = None) -> None:
        """Encode final phase snapshots before archiving or static publishing."""
        if not phase_images:
            return
        with self.condition:
            for phase, image in phase_images.items():
                if image is None:
                    continue
                phase_ok, phase_encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                if phase_ok:
                    self.latest_phase_jpegs[str(phase).lower()] = phase_encoded.tobytes()
            self.condition.notify_all()

    def get_unity_state(self) -> dict:
        try:
            raw = json.loads(self.latest_state.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raw = {}
        summary = raw.get("latest_rep_summary") or {}
        sensor = raw.get("sensor") or {}
        left_pressure = sensor.get("left_pressure_ratio")
        right_pressure = sensor.get("right_pressure_ratio")
        imu_pitch = sensor.get("imu_pitch")
        imu_roll = sensor.get("imu_roll")
        core_pressure = sensor.get("core_pressure_ratio")
        live_quality = raw.get("live_tracking_quality") or "UNKNOWN"
        if isinstance(live_quality, dict):
            # TrackingQualityGate publishes `quality_level`; older clients may
            # still send QUALITY_LEVEL/level/quality. Preserve the canonical
            # value so Unity does not display UNKNOWN for every real frame.
            live_quality = (
                live_quality.get("quality_level")
                or live_quality.get("QUALITY_LEVEL")
                or live_quality.get("level")
                or live_quality.get("quality")
                or "UNKNOWN"
            )
        else:
            match = re.search(r"(?:QUALITY_LEVEL|level|quality)['\"]?\s*[:=]\s*['\"]?([A-Za-z_]+)", str(live_quality))
            if match:
                live_quality = match.group(1)
        last_quality = raw.get("last_rep_quality") or "NONE"
        if isinstance(last_quality, dict):
            last_quality = last_quality.get("level") or last_quality.get("quality") or last_quality.get("overall_quality") or "NONE"
        else:
            match = re.search(r"(?:level|quality|overall_quality)['\"]?\s*[:=]\s*['\"]?([A-Za-z_]+)", str(last_quality))
            if match:
                last_quality = match.group(1)
        live_feedback = raw.get("live_feedback") or describe_live_feedback(raw.get("selected_error") or (raw.get("detected_errors") or [None])[0], raw.get("phase"), raw.get("realtime_feedback"), sensor, raw.get("decision_reason"))
        sensor_mode = str(summary.get("sensor_mode") or sensor.get("source") or "NONE").upper()
        sensor_contract_ok = bool(sensor.get("contract_ok", True))
        sensor_contract_errors = sensor.get("contract_errors") or []
        sensor_contract_invalid = any(token in sensor_mode for token in ("INVALID", "EMPTY", "CONTRACT"))
        sensor_ready = sensor_mode not in {"", "NONE", "NULL"} and sensor_contract_ok and not sensor_contract_invalid
        if sensor_mode == "SERIAL" and sensor_ready:
            sensor_status = "SERIAL SAFELIFT SENSOR READY"
        elif sensor_mode == "MOCK" and sensor_ready:
            sensor_status = "MOCK SENSOR SIMULATOR READY"
        elif sensor_ready:
            sensor_status = f"{sensor_mode} SENSOR SOURCE READY"
        elif sensor_contract_errors:
            sensor_status = "SENSOR CONTRACT INVALID: " + "; ".join(str(item) for item in sensor_contract_errors[:3])
        else:
            sensor_status = "NO SENSOR SOURCE PUBLISHED"

        exception = _unity_exception_payload(
            calibration_status=raw.get("calibration_status") or "measuring",
            live_quality=str(live_quality).upper(),
            sensor_ready=sensor_ready,
            sensor_status=sensor_status,
            report_timeout=bool(raw.get("report_timeout", False)),
            report_status=raw.get("report_status") or "none",
        )

        return {
            "timestamp": raw.get("timestamp", time.time()),
            "phase": raw.get("phase", "Unknown"),
            "rep_count": int(raw.get("rep_count", 0) or 0),
            "set_index": int(raw.get("set_index", 1) or 1),
            "set_rep_count": int(raw.get("set_rep_count", 0) or 0),
            "rep_completed": bool(raw.get("rep_completed", False)),
            "calibration_status": str(raw.get("calibration_status") or "measuring"),
            "calibration_progress": float(raw.get("calibration_progress") or 0.0),
            "calibration_message": str(raw.get("calibration_message") or ""),
            "calibration_baseline": raw.get("calibration_baseline") or {},
            "live_tracking_quality": str(live_quality).upper(),
            "last_rep_quality": str(last_quality).upper(),
            "sensor_mode": sensor_mode,
            "sensor_status": sensor_status,
            "sensor_ready": sensor_ready,
            "exception_code": exception.get("code", "None"),
            "exception_level": exception.get("level", "ok"),
            "exception_message": exception.get("message", ""),
            "current_cue": live_feedback.get("cue") or raw.get("realtime_feedback") or "",
            "live_feedback": live_feedback,
            "live_feedback_issue": live_feedback.get("issue") or "None",
            "live_feedback_body_area": live_feedback.get("body_area") or "none",
            "live_feedback_phase": live_feedback.get("phase") or raw.get("phase") or "Unknown",
            "live_feedback_severity": live_feedback.get("severity") or "none",
            "live_feedback_haptic_channel": live_feedback.get("haptic_channel") or "none",
            "live_feedback_reason": live_feedback.get("reason") or raw.get("decision_reason") or "",
            "detected_errors": raw.get("detected_errors") or [],
            "main_issue": summary.get("main_issue") or raw.get("selected_error") or "None",
            "report_status": raw.get("report_status") or "none",
            "report_elapsed_seconds": float(raw.get("report_elapsed_seconds") or 0.0),
            "report_timeout_seconds": float(raw.get("report_timeout_seconds") or 10.0),
            "report_timeout": bool(raw.get("report_timeout", False)),
            "report_message": raw.get("report_message") or "",
            "report_fallback": bool(raw.get("report_fallback", False)),
            "set_comparison": summary.get("set_comparison") or {},
            "score_summary": summary.get("score_summary") or {},
            "metrics": raw.get("metrics") or {},
            "sensors": {
                "left_pressure_ratio": float(left_pressure or 0.0),
                "right_pressure_ratio": float(right_pressure or 0.0),
                "imu_pitch": float(imu_pitch or 0.0),
                "imu_roll": float(imu_roll or 0.0),
                "core_pressure_ratio": float(core_pressure or 0.0),
                "contract_ok": sensor_contract_ok,
                "contract_errors": sensor_contract_errors,
            },
            "report_source": {"rep_id": summary.get("rep_id", 0), "source": summary.get("report_source", "-"), "reporter": "template", "last_rep_quality": str(last_quality).upper(), "main_issue": summary.get("main_issue") or "None"},
        }

    def update_state(self, state: dict | None = None) -> None:
        if state is None:
            return
        with self.condition:
            safe_state = _json_safe(state)
            self.latest_state = json.dumps(safe_state, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self._archive_report_if_ready(safe_state)
            self._publish_static_report_if_ready(safe_state)
            self.condition.notify_all()

    def get_unity_report(self) -> dict:
        try:
            raw = json.loads(self.latest_state.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raw = {}
        report = raw.get("latest_report") or ""
        summary = raw.get("latest_rep_summary") or {}
        status = str(raw.get("report_status") or "").lower() or ("ready" if report else "none")
        started_at = raw.get("report_started_at")
        timeout_seconds = float(raw.get("report_timeout_seconds") or 10.0)
        elapsed = float(raw.get("report_elapsed_seconds") or 0.0)
        if started_at and status == "generating":
            elapsed = max(elapsed, time.time() - float(started_at))
        timed_out = bool(raw.get("report_timeout", False)) or (status == "generating" and elapsed >= timeout_seconds)
        if timed_out and not report:
            status = "timeout"
        if report:
            status = "ready"
        message = raw.get("report_message") or (
            "AI report is delayed. Showing the basic rule-based analysis first." if timed_out else ""
        )
        return {
            "rep_id": summary.get("rep_id", 0),
            "status": status,
            "report_text": report,
            "elapsed_seconds": round(elapsed, 2),
            "timeout_seconds": timeout_seconds,
            "timed_out": timed_out,
            "fallback": bool(raw.get("report_fallback", False)),
            "message": message,
            "source": raw.get("report_source") or summary.get("report_source") or "-",
        }

    def get_report_page(self, report_id: str | None = None) -> str:
        archived = False
        if report_id:
            raw = self.archived_reports.get(report_id) or self._load_archived_report(report_id) or {}
            archived = bool(raw)
        else:
            try:
                raw = json.loads(self.latest_state.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raw = {}

        report = raw.get("latest_report") or "No AI report has been generated yet. Finish the full SafeLift scenario, then open this page again."
        summary = raw.get("latest_rep_summary") or {}
        rule = summary.get("rule_analysis") or {}
        metrics = summary.get("metrics") or {}
        main_issue = summary.get("main_issue") or rule.get("main_issue") or raw.get("selected_error") or "None"
        detected_errors = summary.get("detected_errors") or rule.get("detected_errors") or raw.get("detected_errors") or []
        phase_quality = summary.get("phase_quality") or (raw.get("last_rep_quality") or {}).get("phase_quality") or {}
        report_source = summary.get("report_source") or raw.get("report_source") or "-"
        timing = summary.get("report_timing") or raw.get("report_timing") or {}
        set_comparison = summary.get("set_comparison") or {}
        feedback_trace = summary.get("feedback_trace") or []
        score_summary = summary.get("score_summary") or {}
        resolved_report_id = report_id if archived else raw.get("latest_report_id") or self.latest_report_id or "latest"
        phase_base_url = f"/api/report_phase_capture/{resolved_report_id}" if archived and resolved_report_id else "/api/phase_capture"

        if archived and report_id:
            self.archived_reports[report_id] = raw

        return _report_html(
            report=report,
            main_issue=main_issue,
            detected_errors=detected_errors,
            phase_quality=phase_quality,
            metrics=metrics,
            report_source=report_source,
            timing=timing,
            set_comparison=set_comparison,
            feedback_trace=feedback_trace,
            score_summary=score_summary,
            report_id=resolved_report_id,
            phase_base_url=phase_base_url,
        )


    def has_archived_report(self, report_id: str | None) -> bool:
        if not report_id:
            return False
        return report_id in self.archived_reports or self._archive_report_path(report_id).exists()

    def get_archived_phase_jpeg(self, report_id: str | None, phase: str) -> bytes | None:
        if not report_id:
            return None
        memory_jpeg = (self.archived_phase_jpegs.get(report_id) or {}).get(phase)
        if memory_jpeg is not None:
            return memory_jpeg
        path = self._archive_phase_path(report_id, phase)
        if path.exists():
            return path.read_bytes()
        return None

    def get_share_report_path(self) -> str:
        if self.latest_static_token:
            # public_report_base already ends with /reports.
            return f"/{self.latest_static_token}/"
        if self.latest_report_id and self.has_archived_report(self.latest_report_id):
            return f"/report/{self.latest_report_id}"
        return "/report/latest"

    def get_share_base_url(self, request_host: str | None = None) -> str:
        if self.static_report_root:
            return self.public_report_base
        return self.get_public_base_url(request_host)

    def _archive_report_if_ready(self, state: dict) -> None:
        if not state or not state.get("latest_report"):
            return
        report_id = self._report_id_from_state(state)
        if not report_id:
            return
        state = dict(state)
        state["latest_report_id"] = report_id
        self.latest_report_id = report_id
        self.archived_reports[report_id] = state
        self.archived_phase_jpegs[report_id] = dict(self.latest_phase_jpegs)
        self._persist_archived_report(report_id, state)

    def _archive_report_path(self, report_id: str) -> Path:
        safe_id = self._safe_archive_id(report_id)
        return self.report_archive_dir / safe_id / "report.json"

    def _archive_phase_path(self, report_id: str, phase: str) -> Path:
        safe_id = self._safe_archive_id(report_id)
        safe_phase = self._safe_archive_id(phase or "phase")
        return self.report_archive_dir / safe_id / "phase" / f"{safe_phase}.jpg"

    def _persist_archived_report(self, report_id: str, state: dict) -> None:
        try:
            report_path = self._archive_report_path(report_id)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
            for phase, jpeg in (self.archived_phase_jpegs.get(report_id) or {}).items():
                phase_path = self._archive_phase_path(report_id, phase)
                phase_path.parent.mkdir(parents=True, exist_ok=True)
                phase_path.write_bytes(jpeg)
        except OSError:
            return

    def _publish_static_report_if_ready(self, state: dict) -> None:
        """Publish a self-contained token report to the Laptop B/Caddy share."""
        if not self.static_report_root or not state.get("latest_report"):
            return
        source_id = self._report_id_from_state(state)
        token = self.static_report_tokens.get(source_id)
        if not token:
            token = secrets.token_urlsafe(12)
            self.static_report_tokens[source_id] = token
        report_dir = self.static_report_root / token
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            (report_dir / "phase").mkdir(parents=True, exist_ok=True)
            raw = dict(state)
            summary = raw.get("latest_rep_summary") or {}
            rule = summary.get("rule_analysis") or {}
            metrics = summary.get("metrics") or {}
            main_issue = summary.get("main_issue") or rule.get("main_issue") or raw.get("selected_error") or "None"
            detected_errors = summary.get("detected_errors") or rule.get("detected_errors") or raw.get("detected_errors") or []
            phase_quality = summary.get("phase_quality") or (raw.get("last_rep_quality") or {}).get("phase_quality") or {}
            report_source = summary.get("report_source") or raw.get("report_source") or "-"
            timing = summary.get("report_timing") or raw.get("report_timing") or {}
            report_html = _report_html(
                report=raw.get("latest_report") or "",
                main_issue=main_issue,
                detected_errors=detected_errors,
                phase_quality=phase_quality,
                metrics=metrics,
                report_source=report_source,
                timing=timing,
                set_comparison=summary.get("set_comparison") or {},
                feedback_trace=summary.get("feedback_trace") or [],
                score_summary=summary.get("score_summary") or {},
                report_id=token,
                phase_base_url="phase",
            )
            (report_dir / "index.html").write_text(report_html, encoding="utf-8")
            for phase, jpeg in (self.latest_phase_jpegs or {}).items():
                (report_dir / "phase" / f"{self._safe_archive_id(phase)}.jpg").write_bytes(jpeg)
            self.latest_static_token = token
            print(f"Published static report: {report_dir / 'index.html'}")
            print(f"Share URL: {self.public_report_base}/{token}/")
        except OSError as exc:
            print(f"Static report publish failed: {exc}")

    def _load_archived_report(self, report_id: str) -> dict | None:
        try:
            report_path = self._archive_report_path(report_id)
            if not report_path.exists():
                return None
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _safe_archive_id(raw: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw or "")).strip("-._")
        return safe[:80] or "latest"

    @staticmethod
    def _report_id_from_state(state: dict) -> str:
        raw = str(state.get("latest_report_id") or state.get("report_id") or "").strip()
        if not raw:
            summary = state.get("latest_rep_summary") or {}
            rep_id = summary.get("rep_id") or state.get("rep_count") or "final"
            started = state.get("report_started_at") or state.get("timestamp") or time.time()
            raw = f"rep{rep_id}_{int(float(started))}"
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-._")
        return safe[:80] or "latest"

    def get_public_base_url(self, request_host: str | None = None) -> str:
        host = (request_host or "").strip()
        if host:
            return f"http://{host}"
        return f"http://{self.host}:{self.port}"

    def get_qr_page(self, request_host: str | None = None) -> str:
        base_url = self.get_share_base_url(request_host)
        report_url = f"{base_url}{self.get_share_report_path()}"
        dashboard_url = f"{base_url}/"
        return _qr_html(
            report_url=report_url,
            qr_png_url=_goqr_image_url(report_url),
            dashboard_url=dashboard_url,
        )

    def get_health(self, request_host: str | None = None) -> dict:
        base_url = self.get_public_base_url(request_host)
        try:
            raw = json.loads(self.latest_state.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raw = {}
        unity_state = self.get_unity_state()
        report = self.get_unity_report()
        calibration = unity_state.get("calibration") or {}
        latest_summary = raw.get("latest_rep_summary") or {}
        set_comparison = latest_summary.get("set_comparison") or {}
        report_ready = str(report.get("status") or "").lower() == "ready" and bool(report.get("report_text"))
        calibration_status = str(calibration.get("status") or unity_state.get("calibration_status") or "unknown")
        calibration_progress = float(calibration.get("progress") or unity_state.get("calibration_progress") or 0.0)
        calibration_baseline = calibration.get("baseline") or unity_state.get("calibration_baseline") or {}
        calibration_samples = int(calibration_baseline.get("sample_count") or 0) if isinstance(calibration_baseline, dict) else 0
        calibration_ready = calibration_status.lower() == "ready" or calibration_progress >= 0.99
        has_set_comparison = bool(set_comparison.get("set1")) and bool(set_comparison.get("set2"))
        sensor_mode = str(unity_state.get("sensor_mode") or raw.get("sensor_mode") or "mock")
        sensor_status = str(unity_state.get("sensor_status") or "unknown")
        sensor_ready = bool(unity_state.get("sensor_ready", False))
        gemini_configured = bool(os.getenv("GEMINI_API_KEY"))
        # Unity may enter Tutorial/Calibration as soon as a backend state is
        # published. Sensor readiness remains a SET 1 gate after calibration.
        unity_launch_ready = bool(raw)
        real_mode_ready = bool(raw) and calibration_ready and sensor_ready
        unity_launch_blockers = []
        if not raw:
            unity_launch_blockers.append("Backend has not published /api/session_state yet.")
        if not sensor_ready:
            unity_launch_blockers.append("Sensor source is not ready yet; Unity can enter Tutorial/Calibration, but SET 1 remains locked.")
        if not unity_launch_blockers:
            unity_launch_blockers.append("None. Unity can enter Tutorial/Calibration; SET 1 remains locked until calibration is ready.")
        real_mode_blockers = []
        if not raw:
            real_mode_blockers.append("Backend has not published /api/session_state yet.")
        if not calibration_ready:
            real_mode_blockers.append("Calibration baseline is not ready.")
        if not sensor_ready:
            real_mode_blockers.append("SafeLift sensor source is not ready. Connect serial/mock provider before starting REAL mode.")
        if not real_mode_blockers:
            real_mode_blockers.append("None. SET 1/SET 2 can run; rep progression remains backend-driven.")
        launch_steps = [
            {"step": "1. Start backend", "ok": bool(self.running), "action": f"Backend must listen on {base_url}."},
            {"step": "2. Publish session state", "ok": bool(raw), "action": "Unity REAL mode reads /api/session_state; keep backend running."},
            {"step": "3. Confirm sensor source", "ok": sensor_ready, "action": "Use the serial/real sensor provider before advancing from Tutorial to SET 1."},
            {"step": "4. Press PLAY for Tutorial", "ok": unity_launch_ready, "action": "Unity may enter Tutorial/Calibration when backend state is available; SET 1 waits for sensor readiness."},
            {"step": "5. Complete calibration", "ok": calibration_ready, "action": "Stand still through tutorial/calibration until SET 1 is unlocked."},
            {"step": "6. Use correct Quest URL", "ok": True, "action": "Editor can use 127.0.0.1; Quest standalone must use the laptop IPv4 URL."},
            {"step": "7. Open laptop QR", "ok": True, "action": "QR is shown at /qr on the laptop browser after the report is ready."},
        ]
        checks = [
            {"name": "Backend HTTP server", "ok": bool(self.running), "detail": f"Listening on {self.host}:{self.port}"},
            {"name": "Unity state endpoint", "ok": bool(raw), "detail": "/api/session_state has current state" if raw else "No state has been published yet"},
            {"name": "Unity REAL Play launch gate", "ok": unity_launch_ready, "detail": "Backend state and sensor provider are ready; Tutorial can start" if unity_launch_ready else "Check unity_launch_blockers"},
            {"name": "SET 1 readiness gate", "ok": real_mode_ready, "detail": "Backend, sensor, and calibration are ready" if real_mode_ready else "Check real_mode_blockers"},
            {"name": "SafeLift sensor source", "ok": sensor_ready, "detail": sensor_status},
            {"name": "Laptop/browser QR handoff", "ok": True, "detail": "QR is shown at /qr in the laptop browser, not as an in-VR QR image"},
            {"name": "Latest report URL", "ok": True, "detail": f"{base_url}/report/latest"},
            {"name": "AI report ready", "ok": report_ready, "detail": str(report.get("status") or "none")},
            {"name": "SET 1/SET 2 comparison", "ok": has_set_comparison, "detail": "Comparison available" if has_set_comparison else "Finish SET 1 and SET 2 first"},
            {"name": "Calibration ready", "ok": calibration_ready, "detail": f"{calibration_status} {calibration_progress:.0%}, samples={calibration_samples}"},
            {"name": "Gemini API key", "ok": gemini_configured, "detail": "GEMINI_API_KEY is set" if gemini_configured else "Template fallback will be used unless GEMINI_API_KEY is set"},
        ]
        integration_checks = [
            {"name": "Unity UI state machine", "ok": True, "out_of_scope": False, "detail": "Start -> Tutorial -> SET1 -> Rest -> SET2 -> Reporting -> Review -> QR flow is implemented."},
            {"name": "Backend session state API", "ok": bool(raw), "out_of_scope": False, "detail": "/api/session_state is publishing current state." if raw else "Start backend/mock session or connect the real provider so state is published."},
            {"name": "Sensor raw acquisition", "ok": False, "out_of_scope": True, "detail": "Explicitly excluded by current user scope. Connect the real Arduino/SafeLift raw provider outside this UI work."},
            {"name": "Sensor provider contract", "ok": sensor_ready, "out_of_scope": False, "detail": sensor_status},
            {"name": "Calibration contract", "ok": calibration_ready, "out_of_scope": False, "detail": f"{calibration_status} {calibration_progress:.0%}, samples={calibration_samples}"},
            {"name": "3+3 squat rep progression", "ok": True, "out_of_scope": False, "detail": "Backend rep counter drives SET 1 three reps and SET 2 three reps before report generation."},
            {"name": "Live feedback contract", "ok": True, "out_of_scope": False, "detail": "SET 2 exposes one selected cue, issue metadata, sensor/pose evidence, and haptic channel."},
            {"name": "Avatar phase replay", "ok": True, "out_of_scope": False, "detail": "Unity records Standing/Descent/Bottom/Ascent snapshots and replays the saved phase in review when an avatar source exists."},
            {"name": "Gemini report contract", "ok": True, "out_of_scope": False, "detail": "Gemini payload and fallback reporter are implemented; live Gemini call requires GEMINI_API_KEY."},
            {"name": "Laptop QR browser handoff", "ok": True, "out_of_scope": False, "detail": "QR is shown on the laptop /qr browser page and links to /report/latest or a fixed report URL."},
            {"name": "Quest physical validation", "ok": False, "out_of_scope": True, "detail": "Requires Meta Quest hardware to verify controller ray, haptics, and Android networking."},
        ]
        ready_without_raw_sensor = bool(raw) and calibration_ready and sensor_ready
        return {
            "ok": bool(self.running),
            "unity_launch_ready": unity_launch_ready,
            "unity_launch_blockers": unity_launch_blockers,
            "ready_for_real_mode_start": real_mode_ready,
            "real_mode_blockers": real_mode_blockers,
            "launch_steps": launch_steps,
            "ready_for_full_review": report_ready and has_set_comparison,
            "sensor_raw_acquisition_in_scope": False,
            "ready_without_raw_sensor": ready_without_raw_sensor,
            "integration_checks": integration_checks,
            "service": "SafeLift XR backend",
            "server": {"host": self.host, "port": self.port, "base_url": base_url},
            "endpoints": {
                "dashboard": f"{base_url}/",
                "health": f"{base_url}/health",
                "qr": f"{base_url}/qr",
                "report": f"{base_url}/report/latest",
                "legacy_report": f"{base_url}/report/demo",
                "session_state": f"{base_url}/api/session_state",
                "latest_report": f"{base_url}/api/latest_report",
                "live_frame": f"{base_url}/api/live_frame.jpg",
            },
            "backend": {
                "running": bool(self.running),
                "has_state": bool(raw),
                "has_live_frame": self.latest_live_jpeg is not None,
                "has_dashboard_frame": self.latest_jpeg is not None,
            },
            "mode": {
                "sensor_mode": sensor_mode,
                "sensor_status": sensor_status,
                "sensor_ready": sensor_ready,
                "unity_launch_ready": unity_launch_ready,
                "real_mode_ready": real_mode_ready,
                "demo_mode_available": True,
            },
            "calibration": {
                "status": calibration_status,
                "progress": calibration_progress,
                "ready": calibration_ready,
                "sample_count": calibration_samples,
                "baseline": calibration_baseline,
            },
            "session": {
                "phase": unity_state.get("phase"),
                "rep_count": unity_state.get("rep_count"),
                "set_index": unity_state.get("set_index"),
                "set_rep_count": unity_state.get("set_rep_count"),
                "set_comparison_ready": has_set_comparison,
            },
            "report": {
                "status": report.get("status"),
                "ready": report_ready,
                "fallback": bool(report.get("fallback")),
                "timed_out": bool(report.get("timed_out")),
                "source": report.get("source"),
            },
            "gemini": {
                "configured": gemini_configured,
                "model": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
                "fallback_expected": not gemini_configured,
            },
            "qr": {
                "browser_handoff": True,
                "in_vr_qr": False,
                "qr_url": f"{base_url}/qr",
                "report_url": f"{base_url}/report/latest",
            },
            "checks": checks,
        }

    def get_health_page(self, request_host: str | None = None) -> str:
        return _health_html(self.get_health(request_host))

    def push_command(self, command: str) -> None:
        with self.command_lock:
            self.pending_commands.append(command)

    def push_unity_rep_score(self, payload: dict) -> bool:
        """Store the Unity/Fusion score for one completed REP.

        Unity is the authoritative scoring client. The Python scorer remains
        available as a fallback when a score packet is missing.
        """
        try:
            set_label = str(payload.get("set") or "").strip().upper()
            rep = int(payload.get("rep") or 0)
            total = float(payload.get("total_score"))
            if set_label not in {"SET 1", "SET 2"} or rep < 1 or total < 0:
                return False
        except (TypeError, ValueError):
            return False

        item = {
            "set": set_label,
            "rep": rep,
            "total_score": total,
            "fsr_score": float(payload.get("fsr_score") or 0.0),
            "fsr_balance_score": float(payload.get("fsr_balance_score") or 0.0),
            "fsr_heel_score": float(payload.get("fsr_heel_score") or 0.0),
            "bracing_score": float(payload.get("bracing_score") or 0.0),
            "spine_score": float(payload.get("spine_score") or 0.0),
            "active_movement_seconds": float(payload.get("active_movement_seconds") or 0.0),
            "balance_duration_seconds": float(payload.get("balance_duration_seconds") or 0.0),
            "heel_duration_seconds": float(payload.get("heel_duration_seconds") or 0.0),
            "bracing_duration_seconds": float(payload.get("bracing_duration_seconds") or 0.0),
            "spine_duration_seconds": float(payload.get("spine_duration_seconds") or 0.0),
            "source": "Unity/Fusion SafeLiftScoring",
        }
        with self.unity_score_lock:
            self.unity_rep_scores[f"{set_label}|{rep}"] = item
        return True

    def get_unity_rep_score(self, set_label: str, rep: int) -> dict | None:
        with self.unity_score_lock:
            item = self.unity_rep_scores.get(f"{set_label}|{int(rep)}")
            return dict(item) if item else None

    def clear_unity_rep_scores(self) -> None:
        with self.unity_score_lock:
            self.unity_rep_scores.clear()

    def pop_command(self) -> str | None:
        with self.command_lock:
            if not self.pending_commands:
                return None
            return self.pending_commands.pop(0)

    def get_latest(self, timeout: float) -> bytes | None:
        with self.condition:
            if self.latest_jpeg is None:
                self.condition.wait(timeout)
            return self.latest_jpeg

    def get_latest_live(self, timeout: float) -> bytes | None:
        with self.condition:
            if self.latest_live_jpeg is None:
                self.condition.wait(timeout)
            return self.latest_live_jpeg

    def stop(self) -> None:
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _goqr_image_url(report_url: str, size: str = "360x360") -> str:
    """Return a goQR/server QR image URL for the final browser report.

    goQR's public image endpoint is intentionally used only by the browser page:
    the backend publishes the final report HTML locally, then the browser loads
    the QR PNG from api.qrserver.com with that report URL as the encoded data.
    """

    encoded = quote(report_url, safe="")
    return f"https://api.qrserver.com/v1/create-qr-code/?size={quote(size, safe='x')}&data={encoded}"


def _report_html(
    report: str,
    main_issue: str,
    detected_errors: list,
    phase_quality: dict,
    metrics: dict,
    report_source: str,
    timing: dict,
    set_comparison: dict | None = None,
    feedback_trace: list | None = None,
    score_summary: dict | None = None,
    report_id: str = "latest",
    phase_base_url: str = "/api/phase_capture",
) -> str:
    safe_report = html.escape(str(report))
    safe_issue = html.escape(str(main_issue))
    safe_errors = html.escape(", ".join(map(str, detected_errors)) if detected_errors else "None")
    safe_source = html.escape(str(report_source))
    elapsed = timing.get("total_seconds") if isinstance(timing, dict) else None
    elapsed_text = f"{float(elapsed):.2f}s" if isinstance(elapsed, (int, float)) else "-"
    score = _score_from_summary(score_summary or {}, str(main_issue), detected_errors)
    phase_rows = _phase_quality_rows(phase_quality)
    score_analysis = (score_summary or {}).get("score_analysis") or {}
    score_analysis_html = _score_analysis_html(score_analysis)
    safe_report_id = html.escape(str(report_id or "latest"))
    safe_phase_base_url = html.escape(str(phase_base_url or "/api/phase_capture"))

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SafeLift XR Report</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; background: #111827; color: #f3f4f6; font-family: Arial, "Malgun Gothic", sans-serif; }}
    header {{ padding: 18px 20px; background: linear-gradient(135deg, #0f172a, #164e63); }}
    h1 {{ margin: 0; font-size: 26px; }}
    .sub {{ color: #bae6fd; margin-top: 6px; }}
    main {{ padding: 18px; display: grid; grid-template-columns: minmax(300px, 1fr) minmax(280px, .8fr); gap: 14px; }}
    section {{ background: rgba(15, 23, 42, .92); border: 1px solid #334155; border-radius: 16px; padding: 16px; box-shadow: 0 18px 42px rgba(0,0,0,.25); }}
    h2 {{ margin: 0 0 12px; color: #67e8f9; font-size: 18px; }}
    pre {{ white-space: pre-wrap; line-height: 1.55; font-family: Arial, "Malgun Gothic", sans-serif; margin: 0; }}
    .score {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; }}
    .card {{ background: #020617; border: 1px solid #1e293b; border-radius: 12px; padding: 12px; }}
    .label {{ color: #94a3b8; font-size: 12px; }}
    .value {{ font-size: 22px; font-weight: 800; margin-top: 4px; }}
    .good {{ color: #86efac; }} .warn {{ color: #fde68a; }} .bad {{ color: #fca5a5; }}
    .captures {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; }}
    .captures img {{ width: 100%; min-height: 120px; object-fit: cover; background: #020617; border: 1px solid #334155; border-radius: 10px; }}
    .avatar-review {{ display: grid; grid-template-columns: minmax(220px,.8fr) minmax(240px,1fr); gap: 14px; align-items: stretch; }}
    .avatar-stage {{ min-height: 340px; background: radial-gradient(circle at 50% 18%, #1e293b, #020617 72%); border: 1px solid #334155; border-radius: 16px; position: relative; overflow: hidden; perspective: 760px; }}
    .avatar-rig {{ position:absolute; inset:0; transform-origin:50% 58%; transition: transform .18s ease; }}
    .body-line {{ position: absolute; left: 50%; top: 70px; width: 4px; height: 170px; background: #94a3b8; transform-origin: top center; border-radius: 999px; transition: transform .18s ease; }}
    .head {{ position: absolute; left: calc(50% - 28px); top: 28px; width: 56px; height: 56px; border-radius: 999px; background: #cbd5e1; box-shadow: 0 0 22px rgba(103,232,249,.25); }}
    .arm, .leg {{ position: absolute; width: 4px; background: #94a3b8; border-radius: 999px; transform-origin: top center; }}
    .arm.left {{ left: calc(50% - 8px); top: 94px; height: 92px; transform: rotate(48deg); }}
    .arm.right {{ left: calc(50% + 8px); top: 94px; height: 92px; transform: rotate(-48deg); }}
    .leg.left {{ left: calc(50% - 2px); top: 232px; height: 96px; transform: rotate(24deg); }}
    .leg.right {{ left: calc(50% + 2px); top: 232px; height: 96px; transform: rotate(-24deg); }}
    .joint {{ position:absolute; width:20px; height:20px; margin:-10px 0 0 -10px; border-radius:999px; background:#38bdf8; border:2px solid #e0f2fe; }}
    .joint.warn {{ background:#f97316; box-shadow:0 0 22px rgba(249,115,22,.85); }}
    .joint.bad {{ background:#ef4444; box-shadow:0 0 24px rgba(239,68,68,.9); }}
    .joint.core {{ width:52px; height:34px; margin:-17px 0 0 -26px; border-radius:18px; }}
    .arrow {{ position:absolute; color:#fbbf24; font-weight:900; font-size:34px; text-shadow:0 2px 10px #000; }}
    .avatar-stage.side .body-line {{ transform: rotate(-12deg) scaleX(.72) !important; }}
    .avatar-stage.side .arm.left {{ transform: rotate(38deg) scaleX(.62); }}
    .avatar-stage.side .arm.right {{ transform: rotate(-22deg) scaleX(.62); opacity:.58; }}
    .avatar-stage.side .leg.left {{ transform: rotate(12deg) scaleX(.62); }}
    .avatar-stage.side .leg.right {{ transform: rotate(-10deg) scaleX(.62); opacity:.58; }}
    .avatar-controls {{ display:flex; flex-wrap:wrap; gap:8px; margin:0 0 10px; }}
    .avatar-controls button, .phase-tabs button {{ cursor:pointer; border:0; color:#cbd5e1; background:#1e293b; padding:7px 10px; border-radius:999px; font-weight:800; font-size:12px; }}
    .avatar-controls button.active, .phase-tabs button.active {{ background:#2563eb; color:#fff; }}
    .avatar-controls button.secondary {{ background:#0f172a; border:1px solid #334155; }}
    .phase-tabs {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }}
    .avatar-note {{ color:#cbd5e1; line-height:1.55; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #334155; padding: 7px 4px; text-align: left; vertical-align: top; }}
    th {{ color: #bae6fd; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>SafeLift XR AI Coach Report</h1>
    <div class="sub">QR 공유 페이지 · AI report + SET 1/SET 2 비교 + phase review</div>
  </header>
  <main>
    <section>
      <h2>AI Report</h2>
      <pre>{safe_report}</pre>
    </section>
    <section>
      <h2>Score Analysis</h2>
      {score_analysis_html}
      <div class="score" style="display:none">
        <div class="card"><div class="label">SafeLift Score</div><div class="value good">{score["safe_lift"]}</div></div>
        <div class="card"><div class="label">FSR Balance /35</div><div class="value good">{score.get("balance", "-")}</div></div>
        <div class="card"><div class="label">Core Bracing /35</div><div class="value good">{score.get("core", "-")}</div></div>
        <div class="card"><div class="label">Spine Alignment /30</div><div class="value good">{score.get("torso", "-")}</div></div>
        <div class="card"><div class="label">Grade</div><div class="value good">{score.get("grade", "-")}</div></div>
        <div class="card"><div class="label">SET 1 → SET 2</div><div class="value good">{score["improvement"]}</div></div>
        <div class="card"><div class="label">Main Issue</div><div class="value warn">{safe_issue}</div></div>
        <div class="card"><div class="label">Detected Errors</div><div class="value bad">{safe_errors}</div></div>
      </div>
    </section>
    <section>
      <h2>Phase Captures</h2>
      <div class="captures">
        <div><img src="{safe_phase_base_url}/standing.jpg" alt="standing"><div class="label">Standing</div></div>
        <div><img src="{safe_phase_base_url}/descent.jpg" alt="descent"><div class="label">Descent</div></div>
        <div><img src="{safe_phase_base_url}/bottom.jpg" alt="bottom"><div class="label">Bottom</div></div>
        <div><img src="{safe_phase_base_url}/ascent.jpg" alt="ascent"><div class="label">Ascent</div></div>
      </div>
    </section>
  </main>
  <script>
    (function () {{
      var stage = document.getElementById('browserAvatarStage');
      var rig = document.getElementById('browserAvatarRig');
      var selectedPhase = document.getElementById('browserAvatarSelectedPhase');
      var phaseCue = document.getElementById('browserAvatarPhaseCue');
      var rotation = 0;
      var zoom = 1;
      var poseStyles = {{
        Standing: {{ body:'rotate(0deg)', leftArm:'rotate(16deg)', rightArm:'rotate(-16deg)', leftLeg:'rotate(5deg)', rightLeg:'rotate(-5deg)', cue:'준비 자세: 발 전체로 바닥을 누르고 기본 정렬을 확인하세요.' }},
        Descent: {{ body:'rotate(-7deg)', leftArm:'rotate(34deg)', rightArm:'rotate(-34deg)', leftLeg:'rotate(25deg)', rightLeg:'rotate(-25deg)', cue:'내려가기: 무릎과 엉덩이를 함께 굽히며 천천히 내려가세요.' }},
        Bottom: {{ body:'rotate(-11deg)', leftArm:'rotate(58deg)', rightArm:'rotate(-58deg)', leftLeg:'rotate(54deg)', rightLeg:'rotate(-54deg)', cue:'가장 낮은 자세: 안정적으로 유지할 수 있는 깊이까지만 내려가세요.' }},
        Ascent: {{ body:'rotate(-5deg)', leftArm:'rotate(42deg)', rightArm:'rotate(-42deg)', leftLeg:'rotate(30deg)', rightLeg:'rotate(-30deg)', cue:'올라오기: 양발로 바닥을 밀고 몸통을 단단히 유지하세요.' }}
      }};
      function applyRigTransform() {{
        if (!rig) return;
        rig.style.transform = 'rotateY(' + rotation + 'deg) scale(' + zoom.toFixed(2) + ')';
      }}
      function setAvatarPhase(phase) {{
        var pose = poseStyles[phase] || poseStyles.Standing;
        var body = document.getElementById('browserAvatarBody');
        var leftArm = document.getElementById('browserAvatarLeftArm');
        var rightArm = document.getElementById('browserAvatarRightArm');
        var leftLeg = document.getElementById('browserAvatarLeftLeg');
        var rightLeg = document.getElementById('browserAvatarRightLeg');
        if (body) body.style.transform = pose.body;
        if (leftArm) leftArm.style.transform = pose.leftArm;
        if (rightArm) rightArm.style.transform = pose.rightArm;
        if (leftLeg) leftLeg.style.transform = pose.leftLeg;
        if (rightLeg) rightLeg.style.transform = pose.rightLeg;
        if (selectedPhase) selectedPhase.textContent = phase;
        if (phaseCue) phaseCue.textContent = pose.cue;
      }}
      document.querySelectorAll('[data-avatar-view]').forEach(function (button) {{
        button.addEventListener('click', function () {{
          document.querySelectorAll('[data-avatar-view]').forEach(function (item) {{ item.classList.remove('active'); }});
          button.classList.add('active');
          if (!stage) return;
          stage.classList.toggle('side', button.getAttribute('data-avatar-view') === 'side');
        }});
      }});
      document.querySelectorAll('[data-avatar-action]').forEach(function (button) {{
        button.addEventListener('click', function () {{
          var action = button.getAttribute('data-avatar-action');
          if (action === 'rotate-left') rotation -= 18;
          if (action === 'rotate-right') rotation += 18;
          if (action === 'zoom-in') zoom = Math.min(1.45, zoom + .12);
          if (action === 'zoom-out') zoom = Math.max(.75, zoom - .12);
          if (action === 'reset') {{ rotation = 0; zoom = 1; }}
          applyRigTransform();
        }});
      }});
      document.querySelectorAll('.phase-tabs [data-phase]').forEach(function (button) {{
        button.addEventListener('click', function () {{
          document.querySelectorAll('.phase-tabs [data-phase]').forEach(function (item) {{ item.classList.remove('active'); }});
          button.classList.add('active');
          setAvatarPhase(button.getAttribute('data-phase'));
        }});
      }});
      var defaultPhase = stage ? stage.getAttribute('data-default-phase') : 'Standing';
      setAvatarPhase(defaultPhase || 'Standing');
      applyRigTransform();
    }})();
  </script>
</body>
</html>"""


def _score_analysis_html(analysis: dict) -> str:
    if not isinstance(analysis, dict) or not analysis.get("rep_scores"):
        return '<p class="label">REP별 점수 분석은 SET 1과 SET 2의 3회 반복이 완료되면 표시됩니다.</p>'

    def value(item) -> str:
        if item is None:
            return "-"
        if isinstance(item, float):
            return f"{item:.1f}"
        return html.escape(str(item))

    rows = []
    for item in analysis.get("rep_scores") or []:
        score = item.get("score_summary") or {}
        rubric = score.get("rubric") or {}
        balance = rubric.get("ground_balance_fsr") or {}
        deductions = item.get("deductions") or {}
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('set', '-')))} REP {value(item.get('rep'))}</td>"
            f"<td><b>{value(item.get('score'))}/100</b></td>"
            f"<td>{value(score.get('balance_score'))}/35 (-{value(deductions.get('fsr_balance'))})</td>"
            f"<td>{value(balance.get('heel_maintain_score', balance.get('forefoot_stability_score')))} / 12 (-{value(deductions.get('heel_maintenance'))})</td>"
            f"<td>{value(score.get('spine_alignment_score'))}/30 (-{value(deductions.get('spine_alignment'))})</td>"
            f"<td>{value(score.get('core_bracing_score'))}/35 (-{value(deductions.get('core_bracing'))})</td>"
            "</tr>"
        )

    final_best = analysis.get("final_best") or {}
    final_summary = final_best.get("score_summary") or {}
    return f"""
      <table>
        <thead><tr><th>REP</th><th>Total</th><th>좌/우 밸런스</th><th>뒤꿈치 유지</th><th>허리/척추 정렬</th><th>브레이싱 유지</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
      <div class="score">
        <div class="card"><div class="label">Final Score (6 REP 중 최고점)</div><div class="value good">{value(final_best.get('score'))}/100</div></div>
        <div class="card"><div class="label">Final Grade</div><div class="value good">{value(final_summary.get('grade'))}</div></div>
        <div class="card"><div class="label">Selected REP</div><div class="value good">{html.escape(str(final_best.get('set', '-')))} REP {value(final_best.get('rep'))}</div></div>
      </div>
      <p class="label">{html.escape(str(analysis.get('improvement_message') or ''))}</p>
    """


def _set_comparison_html(comparison: dict, score_analysis: dict | None = None) -> str:
    analysis = score_analysis or {}
    set1 = analysis.get("set1_best") or {}
    set2 = analysis.get("set2_best") or {}
    if not set1 or not set2:
        return '<p class="label">SET 1과 SET 2의 최고점 비교는 6회 REP 분석이 완료되면 표시됩니다.</p>'

    def value(item) -> str:
        if item is None:
            return "-"
        if isinstance(item, float):
            return f"{item:.1f}"
        return html.escape(str(item))

    def row(label: str, key: str, maximum: int) -> str:
        s1 = set1.get("score_summary") or {}
        s2 = set2.get("score_summary") or {}
        delta = (s2.get(key) or 0) - (s1.get(key) or 0)
        return (
            f"<tr><td>{html.escape(label)}</td>"
            f"<td>{value(s1.get(key))}/{maximum}</td>"
            f"<td>{value(s2.get(key))}/{maximum}</td>"
            f"<td>{value(delta)}</td></tr>"
        )

    return f"""
      <p class="label">SET 1 최고점: {value(set1.get('score'))}/100 (REP {value(set1.get('rep'))}) · SET 2 최고점: {value(set2.get('score'))}/100 (REP {value(set2.get('rep'))})</p>
      <table>
        <thead><tr><th>채점 항목</th><th>SET 1 최고 REP</th><th>SET 2 최고 REP</th><th>변화</th></tr></thead>
        <tbody>
          {row("좌/우 FSR 밸런스", "balance_score", 35)}
          {row("허리/척추 정렬", "spine_alignment_score", 30)}
          {row("브레이싱 유지", "core_bracing_score", 35)}
        </tbody>
      </table>
      <p class="label">{html.escape(str(analysis.get('improvement_message') or ''))}</p>
    """


def _legacy_set_comparison_html(comparison: dict) -> str:
    if not isinstance(comparison, dict) or not comparison:
        return '<p class="label">SET comparison has not been captured yet. Finish SET 1 and SET 2 to populate this table.</p>'

    set1 = comparison.get("set1") or {}
    set2 = comparison.get("set2") or {}
    metrics1 = set1.get("metrics") or {}
    metrics2 = set2.get("metrics") or {}

    def value(item) -> str:
        if item is None:
            return "-"
        if isinstance(item, float):
            return f"{item:.3f}"
        return html.escape(str(item))

    rows = [
        ("Frames", set1.get("frame_count"), set2.get("frame_count"), None),
        ("Rule issue count", set1.get("error_count"), set2.get("error_count"), comparison.get("error_count_delta")),
        ("Live cue count", metrics1.get("feedback_count"), metrics2.get("feedback_count"), comparison.get("feedback_count_delta")),
        ("Torso peak angle", metrics1.get("torso_angle_peak_abs"), metrics2.get("torso_angle_peak_abs"), comparison.get("torso_angle_peak_abs_delta")),
        ("Knee valgus peak", metrics1.get("knee_valgus_peak"), metrics2.get("knee_valgus_peak"), comparison.get("knee_valgus_peak_delta")),
    ]
    row_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(label)}</td>"
        f"<td>{value(before)}</td>"
        f"<td>{value(after)}</td>"
        f"<td>{value(delta)}</td>"
        "</tr>"
        for label, before, after, delta in rows
    )
    summary = html.escape(str(comparison.get("summary") or "SET comparison summary is unavailable."))
    set1_errors = html.escape(", ".join(map(str, set1.get("detected_errors") or [])) or "None")
    set2_errors = html.escape(", ".join(map(str, set2.get("detected_errors") or [])) or "None")

    return f"""
      <p class="label">{summary}</p>
      <table>
        <thead><tr><th>Metric</th><th>SET 1</th><th>SET 2</th><th>Delta</th></tr></thead>
        <tbody>{row_html}</tbody>
      </table>
      <p class="label">SET 1 issues: {set1_errors}<br>SET 2 issues: {set2_errors}</p>
    """


def _feedback_trace_html(trace: list) -> str:
    if not isinstance(trace, list) or not trace:
        return '<p class="label">No live feedback trace has been captured yet. SET 2 feedback decisions will appear here after the full scenario.</p>'

    def value(item) -> str:
        if item is None or item == "":
            return "-"
        if isinstance(item, float):
            return f"{item:.3f}"
        return html.escape(str(item))

    rows = []
    for item in trace[-8:]:
        if not isinstance(item, dict):
            continue
        sensor = item.get("sensor") or {}
        metrics = item.get("metrics") or {}
        sensor_text = (
            f"L {value(sensor.get('left_pressure_ratio'))} / "
            f"R {value(sensor.get('right_pressure_ratio'))} / "
            f"Core {value(sensor.get('core_pressure_ratio'))} / "
            f"IMU {value(sensor.get('imu_pitch'))},{value(sensor.get('imu_roll'))}"
        )
        metric_text = (
            f"Knee {value(metrics.get('avg_knee_angle'))} / "
            f"Torso {value(metrics.get('torso_angle'))} / "
            f"Valgus {value(metrics.get('knee_valgus_score'))}"
        )
        rows.append(
            "<tr>"
            f"<td>SET {value(item.get('set_index'))}<br>Rep {value(item.get('set_rep_count'))}</td>"
            f"<td>{value(item.get('phase'))}</td>"
            f"<td>{value(item.get('issue'))}</td>"
            f"<td>{value(item.get('cue'))}</td>"
            f"<td>{value(item.get('decision_reason'))}</td>"
            f"<td>{sensor_text}<br>{metric_text}</td>"
            "</tr>"
        )
    if not rows:
        return '<p class="label">Feedback trace payload exists, but no renderable entries were found.</p>'
    return (
        "<table>"
        "<thead><tr><th>Set/Rep</th><th>Phase</th><th>Issue</th><th>Cue</th><th>Decision Reason</th><th>Sensor / Pose Evidence</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _qr_html(report_url: str, qr_png_url: str, dashboard_url: str) -> str:
    safe_report_url = html.escape(report_url)
    safe_qr_png_url = html.escape(qr_png_url)
    safe_dashboard_url = html.escape(dashboard_url)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SafeLift XR QR</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin: 0; min-height: 100vh; display: grid; place-items: center; background: radial-gradient(circle at top, #164e63, #020617 62%); color: #f8fafc; font-family: Arial, "Malgun Gothic", sans-serif; }}
    main {{ width: min(92vw, 760px); background: rgba(15, 23, 42, .94); border: 1px solid #334155; border-radius: 28px; padding: 34px; box-shadow: 0 28px 80px rgba(0,0,0,.38); text-align: center; }}
    h1 {{ margin: 0; font-size: clamp(28px, 4vw, 44px); color: #67e8f9; }}
    .sub {{ margin: 12px auto 26px; max-width: 560px; color: #cbd5e1; line-height: 1.55; font-size: 17px; }}
    .qr {{ display: inline-block; background: #fff; padding: 20px; border-radius: 22px; box-shadow: 0 18px 46px rgba(0,0,0,.32); }}
    .qr img {{ width: min(64vw, 360px); height: min(64vw, 360px); display: block; }}
    .url {{ margin: 22px auto 0; padding: 12px 14px; border-radius: 14px; background: #020617; border: 1px solid #1e293b; color: #bae6fd; word-break: break-all; font-size: 15px; }}
    .actions {{ margin-top: 22px; display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; }}
    a {{ color: white; text-decoration: none; font-weight: 800; padding: 12px 16px; border-radius: 999px; background: #2563eb; }}
    a.secondary {{ background: #334155; }}
    .note {{ margin-top: 20px; color: #fde68a; line-height: 1.55; font-size: 15px; }}
    code {{ color: #bae6fd; }}
  </style>
</head>
<body>
  <main>
    <h1>SafeLift XR 노트북 QR</h1>
    <p class="sub">이 화면은 VR 안이 아니라 노트북 브라우저에 띄우는 QR 공유 화면입니다. HMD를 벗고 휴대폰으로 아래 QR을 스캔하면 AI 리포트와 SET 1/SET 2 비교 결과를 볼 수 있습니다.</p>
    <div class="qr"><img src="{safe_qr_png_url}" alt="SafeLift report QR code generated by goQR API"></div>
    <div class="url">{safe_report_url}</div>
    <div class="actions">
      <a href="{safe_report_url}">AI 리포트 열기</a>
      <a class="secondary" href="{safe_dashboard_url}">대시보드로 돌아가기</a>
    </div>
    <div class="note">노트북에서만 확인할 때는 <code>http://127.0.0.1:8090/qr</code><br>휴대폰으로 스캔할 때는 <code>http://&lt;노트북_IP&gt;:8090/qr</code> 주소로 이 페이지를 여세요.</div>
  </main>
  <script>
    (function () {{
      var currentReportUrl = {json.dumps(report_url, ensure_ascii=False)};
      setInterval(function () {{
        fetch('/api/qr_link', {{ cache: 'no-store' }})
          .then(function (response) {{ return response.json(); }})
          .then(function (payload) {{
            if (payload.url && payload.url !== currentReportUrl) window.location.reload();
          }})
          .catch(function () {{ /* backend may be briefly busy while saving the report */ }});
      }}, 3000);
    }})();
  </script>
</body>
</html>"""

def _health_html(health: dict) -> str:
    endpoints = health.get("endpoints") or {}
    checks = health.get("checks") or []

    def yn(value: bool) -> str:
        return "OK" if value else "WAIT"

    check_rows = "".join(
        f"<tr><td>{html.escape(str(item.get('name')))}</td><td class='{('ok' if item.get('ok') else 'wait')}'>{yn(bool(item.get('ok')))}</td><td>{html.escape(str(item.get('detail') or '-'))}</td></tr>"
        for item in checks
    )
    endpoint_rows = "".join(
        f"<tr><td>{html.escape(str(key))}</td><td><a href='{html.escape(str(url))}'>{html.escape(str(url))}</a></td></tr>"
        for key, url in endpoints.items()
    )
    qr = health.get("qr") or {}
    report = health.get("report") or {}
    mode = health.get("mode") or {}
    gemini = health.get("gemini") or {}
    blockers = health.get("real_mode_blockers") or []
    launch_steps = health.get("launch_steps") or []
    integration_checks = health.get("integration_checks") or []
    blocker_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in blockers)
    launch_rows = "".join(
        f"<tr><td>{html.escape(str(item.get('step')))}</td><td class='{('ok' if item.get('ok') else 'wait')}'>{yn(bool(item.get('ok')))}</td><td>{html.escape(str(item.get('action') or '-'))}</td></tr>"
        for item in launch_steps
    )
    integration_rows = "".join(
        f"<tr><td>{html.escape(str(item.get('name')))}</td><td class='{('wait' if item.get('out_of_scope') else ('ok' if item.get('ok') else 'wait'))}'>{'N/A' if item.get('out_of_scope') else yn(bool(item.get('ok')))}</td><td>{html.escape(str(item.get('detail') or '-'))}</td></tr>"
        for item in integration_checks
    )
    backend_class = "okBg" if health.get("ok") else "waitBg"
    unity_launch_class = "okBg" if health.get("unity_launch_ready") else "waitBg"
    real_class = "okBg" if health.get("ready_for_real_mode_start") else "waitBg"
    review_class = "okBg" if health.get("ready_for_full_review") else "waitBg"
    qr_url = html.escape(str(qr.get("qr_url") or "/qr"))
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SafeLift XR Health Check</title>
  <style>
    :root {{ color-scheme: dark; }}
    body {{ margin:0; background:#020617; color:#f8fafc; font-family:Arial, "Malgun Gothic", sans-serif; }}
    header {{ padding:22px; background:linear-gradient(135deg,#0f172a,#155e75); }}
    main {{ padding:18px; display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }}
    section {{ background:#0f172a; border:1px solid #334155; border-radius:18px; padding:16px; box-shadow:0 18px 42px rgba(0,0,0,.25); }}
    section.wide {{ grid-column:1/-1; }}
    h1 {{ margin:0; font-size:28px; }} h2 {{ margin:0 0 10px; color:#67e8f9; font-size:18px; }}
    .pill {{ display:inline-block; padding:6px 10px; border-radius:999px; font-weight:800; margin-right:6px; }}
    .ok {{ color:#86efac; }} .wait {{ color:#fde68a; }} .bad {{ color:#fca5a5; }}
    .okBg {{ background:#14532d; }} .waitBg {{ background:#713f12; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }} th,td {{ border-bottom:1px solid #334155; padding:8px 4px; text-align:left; vertical-align:top; }}
    a {{ color:#93c5fd; word-break:break-all; }} .note {{ color:#cbd5e1; line-height:1.55; }}
  </style>
</head>
<body>
  <header>
    <h1>SafeLift XR Health Check</h1>
    <p class="note">Pre-run status for backend, Real/Demo mode, Gemini, report, and laptop-browser QR sharing.</p>
  </header>
  <main>
    <section><h2>Overall</h2>
      <span class="pill {backend_class}">Backend {yn(bool(health.get('ok')))}</span>
      <span class="pill {unity_launch_class}">Unity Play {yn(bool(health.get('unity_launch_ready')))}</span>
      <span class="pill {real_class}">SET 1 Ready {yn(bool(health.get('ready_for_real_mode_start')))}</span>
      <span class="pill {review_class}">Full Review {yn(bool(health.get('ready_for_full_review')))}</span>
      <p class="note">The QR code is intentionally shown on the laptop browser <a href="{qr_url}">/qr</a> page, not as an in-VR QR image.</p>
    </section>
    <section class="wide"><h2>Real Mode Launch Checklist</h2>
      <p class="note">REAL mode has two gates: Unity Play requires backend state plus a usable SafeLift sensor provider, then SET 1 remains locked until Tutorial/Calibration completes. Sensor raw acquisition remains outside this Unity UI scope, but the provider readiness gate is explicit here.</p>
      <table><thead><tr><th>Step</th><th>Status</th><th>Action</th></tr></thead><tbody>{launch_rows}</tbody></table>
      <p class="note"><b>Current blockers:</b></p><ul class="note">{blocker_items}</ul>
    </section>
    <section class="wide"><h2>Integration Contract</h2>
      <p class="note">This table separates implemented SafeLift XR contracts from work that is intentionally outside this scope. Sensor raw acquisition is marked N/A here because the current task excludes reading raw hardware values; the provider contract is still checked.</p>
      <table><thead><tr><th>Contract</th><th>Status</th><th>Detail</th></tr></thead><tbody>{integration_rows}</tbody></table>
    </section>
    <section><h2>Mode / Report</h2>
      <table><tbody>
        <tr><td>Sensor mode</td><td>{html.escape(str(mode.get('sensor_mode')))}</td></tr>
        <tr><td>Sensor status</td><td>{html.escape(str(mode.get('sensor_status')))} / ready={yn(bool(mode.get('sensor_ready')))}</td></tr>
        <tr><td>Report status</td><td>{html.escape(str(report.get('status')))}</td></tr>
        <tr><td>Report source</td><td>{html.escape(str(report.get('source')))}</td></tr>
        <tr><td>Gemini configured</td><td>{yn(bool(gemini.get('configured')))} - {html.escape(str(gemini.get('model')))}</td></tr>
        <tr><td>In-VR QR image</td><td>{str(bool(qr.get('in_vr_qr')))} - intended value is False</td></tr>
      </tbody></table>
    </section>
    <section><h2>Checks</h2><table><thead><tr><th>Check</th><th>Status</th><th>Detail</th></tr></thead><tbody>{check_rows}</tbody></table></section>
    <section><h2>Endpoints</h2><table><tbody>{endpoint_rows}</tbody></table></section>
  </main>
</body>
</html>"""


def _browser_avatar_review_html(main_issue: str, detected_errors: list) -> str:
    text = " ".join([main_issue or "", " ".join(map(str, detected_errors or []))])
    issue = "None"
    phase = "Standing"
    cue = "전체 정렬이 안정적입니다. 같은 템포를 유지하세요."
    highlights = {
        "left_knee": "joint",
        "right_knee": "joint",
        "torso": "joint",
        "core": "joint core",
        "left_foot": "joint",
        "right_foot": "joint",
    }
    arrow_html = ""

    if "KneeValgus" in text:
        issue = "Knee Valgus"
        phase = "Ascent"
        cue = "무릎이 안쪽으로 모이지 않게 발끝 방향으로 밀어주세요."
        highlights["left_knee"] = "joint bad"
        highlights["right_knee"] = "joint bad"
        arrow_html = '<div class="arrow" style="left:34%;top:206px">↔</div><div class="arrow" style="right:31%;top:206px">↔</div>'
    elif "WeightShift" in text or "WeightAsymmetry" in text:
        issue = "Weight Shift"
        phase = "Bottom"
        cue = "좌우 발 압력을 가운데로 맞추고 midfoot에 무게를 유지하세요."
        highlights["left_foot"] = "joint warn"
        highlights["right_foot"] = "joint warn"
        arrow_html = '<div class="arrow" style="left:45%;top:252px">↔</div>'
    elif "CorePressureDrop" in text:
        issue = "Core Bracing"
        phase = "Descent"
        cue = "내려가기 전에 복압을 다시 만들고 하강 중 유지하세요."
        highlights["core"] = "joint core bad"
    elif "TorsoLean" in text or "ShallowDepth" in text or "Complex" in text or "TrackingUncertain" in text:
        issue = "Torso / Depth"
        phase = "Bottom"
        cue = "상체 각도와 깊이를 천천히 확인하고 흔들림을 줄이세요."
        highlights["torso"] = "joint bad"
        arrow_html = '<div class="arrow" style="left:56%;top:104px">↙</div>'

    phase_buttons = "".join(
        f'<button type="button" data-phase="{html.escape(item)}" class="{("active" if item == phase else "")}">{html.escape(item)}</button>'
        for item in ("Standing", "Descent", "Bottom", "Ascent")
    )

    return f"""
      <div class="avatar-review" data-default-phase="{html.escape(phase)}">
        <div class="avatar-stage" id="browserAvatarStage" data-default-phase="{html.escape(phase)}" aria-label="Interactive browser avatar phase review">
          <div class="avatar-rig" id="browserAvatarRig">
            <div class="head"></div>
            <div class="body-line" id="browserAvatarBody"></div>
            <div class="arm left" id="browserAvatarLeftArm"></div><div class="arm right" id="browserAvatarRightArm"></div>
            <div class="leg left" id="browserAvatarLeftLeg"></div><div class="leg right" id="browserAvatarRightLeg"></div>
            <div class="{highlights['torso']}" style="left:50%;top:118px"></div>
            <div class="{highlights['core']}" style="left:50%;top:176px"></div>
            <div class="{highlights['left_knee']}" style="left:39%;top:247px"></div>
            <div class="{highlights['right_knee']}" style="left:61%;top:247px"></div>
            <div class="{highlights['left_foot']}" style="left:33%;top:319px"></div>
            <div class="{highlights['right_foot']}" style="left:67%;top:319px"></div>
            {arrow_html}
          </div>
        </div>
        <div>
          <div class="avatar-controls" aria-label="Avatar view controls">
            <button type="button" class="active" data-avatar-view="front">Front View</button>
            <button type="button" data-avatar-view="side">Side View</button>
            <button type="button" class="secondary" data-avatar-action="rotate-left">Rotate ◀</button>
            <button type="button" class="secondary" data-avatar-action="rotate-right">Rotate ▶</button>
            <button type="button" class="secondary" data-avatar-action="zoom-in">Zoom +</button>
            <button type="button" class="secondary" data-avatar-action="zoom-out">Zoom -</button>
            <button type="button" class="secondary" data-avatar-action="reset">Reset</button>
          </div>
          <div class="phase-tabs" aria-label="Phase tabs">
            {phase_buttons}
          </div>
          <div class="card">
            <div class="label">Selected Review Phase</div>
            <div class="value warn" id="browserAvatarSelectedPhase">{html.escape(phase)}</div>
          </div>
          <div class="card" style="margin-top:10px">
            <div class="label">Highlighted Issue</div>
            <div class="value bad">{html.escape(issue)}</div>
          </div>
          <p class="avatar-note">이 브라우저 avatar review는 QR report에서 바로 확인하는 공유용 phase review입니다. phase tab을 누르면 준비 자세, 내려가기, 가장 낮은 자세, 올라오기 pose가 바뀌고 Rotate/Zoom으로 방향과 크기를 조절할 수 있습니다. HMD 안에서는 Unity가 저장한 humanoid phase pose를 더 자세히 회전·확대해서 확인합니다.</p>
          <p class="avatar-note"><b>Phase cue:</b> <span id="browserAvatarPhaseCue">{html.escape(cue)}</span></p>
          <p class="avatar-note"><b>Issue cue:</b> {html.escape(cue)}</p>
        </div>
      </div>
    """

def _score_from_summary(score_summary: dict, main_issue: str, detected_errors: list) -> dict:
    if isinstance(score_summary, dict) and score_summary.get("safe_lift_score") is not None:
        improvement = score_summary.get("improvement_delta")
        if isinstance(improvement, (int, float)):
            improvement_text = f"{improvement:+.0f} rubric"
        else:
            improvement_text = str(score_summary.get("summary") or "score payload")
        grade = str(score_summary.get("grade", "-"))
        if score_summary.get("grade_label"):
            grade += " · " + str(score_summary.get("grade_label"))
        return {
            "safe_lift": str(score_summary.get("safe_lift_score")),
            "improvement": improvement_text,
            "knee": str(score_summary.get("knee_alignment_score", "-")),
            "balance": str(score_summary.get("balance_score", "-")),
            "torso": str(score_summary.get("spine_alignment_score", score_summary.get("torso_stability_score", "-"))),
            "core": str(score_summary.get("core_bracing_score", "-")),
            "grade": grade,
            "evidence": _score_evidence_label(score_summary),
        }
    return _score_from_issue(main_issue, detected_errors)


def _score_evidence_label(score_summary: dict) -> str:
    if not isinstance(score_summary, dict):
        return "-"
    if score_summary.get("score_evidence_complete") is True:
        return "complete"
    missing = score_summary.get("missing_score_evidence") or []
    if missing:
        return "partial: " + ", ".join(map(str, missing[:4]))
    return str(score_summary.get("score_confidence") or "partial")


def _score_from_issue(main_issue: str, detected_errors: list) -> dict:
    text = " ".join([main_issue or "", " ".join(map(str, detected_errors or []))])
    if "Complex" in text or "TorsoLean" in text or "ShallowDepth" in text or "TrackingUncertain" in text:
        return {"safe_lift": "76", "improvement": "+8"}
    if "CorePressureDrop" in text:
        return {"safe_lift": "82", "improvement": "+15 core"}
    if "WeightShift" in text:
        return {"safe_lift": "84", "improvement": "+9 balance"}
    if "KneeValgus" in text:
        return {"safe_lift": "84", "improvement": "+12 knee"}
    return {"safe_lift": "90", "improvement": "+3 stable"}


def _phase_quality_rows(phase_quality: dict) -> str:
    rows = []
    for phase in ("standing", "descent", "bottom", "ascent"):
        item = phase_quality.get(phase, {}) if isinstance(phase_quality, dict) else {}
        quality = html.escape(str(item.get("quality") or item.get("level") or "-"))
        frames = html.escape(str(item.get("frame_count") or 0))
        reason = html.escape(str(item.get("reason") or "-"))
        rows.append(f"<tr><td>{phase}</td><td>{quality}</td><td>{frames}</td><td>{reason}</td></tr>")
    return "".join(rows)


def _placeholder_phase_jpeg(phase: str) -> bytes:
    label = (phase or "phase").replace("_", " ").title()
    canvas = np.full((360, 560, 3), (15, 23, 42), dtype=np.uint8)
    cv2.rectangle(canvas, (18, 18), (542, 342), (51, 65, 85), 2)
    cv2.putText(canvas, "SafeLift XR", (38, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.0, CYAN, 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (38, 148), cv2.FONT_HERSHEY_SIMPLEX, 1.4, WHITE, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Phase capture pending", (38, 215), cv2.FONT_HERSHEY_SIMPLEX, .85, YELLOW, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Complete a rep to replace this image.", (38, 265), cv2.FONT_HERSHEY_SIMPLEX, .62, WHITE, 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        return b""
    return encoded.tobytes()


def _browser_html() -> str:
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SafeLift XR</title>
  <style>
    :root { color-scheme: dark; }
    body { margin: 0; background: #151515; color: #eeeeee; font-family: Arial, "Malgun Gothic", sans-serif; }
    header { padding: 10px 14px; background: #222; font-weight: 700; }
    main { display: grid; grid-template-columns: minmax(520px, 1.45fr) minmax(360px, .75fr); gap: 14px; padding: 14px; }
    img { width: 100%; height: auto; background: #0c0c0c; border: 1px solid #333; }
    section { background: #202020; border: 1px solid #333; padding: 12px; }
    h2 { margin: 0 0 10px; font-size: 17px; color: #d8eef2; }
    button { background: #2d6cdf; color: white; border: 0; padding: 10px 12px; font-weight: 700; cursor: pointer; }
    button.secondary { background: #444; }
    button:active { transform: translateY(1px); }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .item { padding: 8px; background: #181818; border: 1px solid #303030; }
    .label { color: #aaaaaa; font-size: 12px; margin-bottom: 4px; }
    .value { font-size: 18px; line-height: 1.35; word-break: keep-all; }
    .hint { color: #bbbbbb; font-size: 13px; line-height: 1.45; margin: 8px 0 10px; }
    pre { white-space: pre-wrap; margin: 0; line-height: 1.5; font-family: Arial, "Malgun Gothic", sans-serif; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; line-height: 1.35; }
    th, td { border: 1px solid #333; padding: 6px; text-align: left; vertical-align: top; }
    th { background: #181818; color: #d8eef2; }
    td.reason { color: #dddddd; }
    .feedback { color: #ffe780; }
    .error { color: #ff8d8d; }
    .evidence { color: #dddddd; font-size: 14px; }
    @media (max-width: 980px) { main { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <header>SafeLift XR browser dashboard</header>
  <main>
    <div><img src="/stream.mjpg" alt="webcam dashboard stream"></div>
    <div>
      <section>
        <h2>Live State</h2>
        <div class="actions">
          <button onclick="sendCommand('finish_rep')">Generate Final Report after 10 reps</button>
          <button class="secondary" onclick="sendCommand('reset_rep')">Reset Rep</button>
          <button class="secondary" onclick="sendCommand('save_snapshot')">Save Snapshot</button>
          <button class="secondary" onclick="window.open('/qr', '_blank')">Open Laptop QR</button>
          <button class="secondary" onclick="window.open('/health', '_blank')">Health Check</button>
        </div>
        <div id="commandStatus" class="hint">Command status: ready</div>
        <div class="hint">Browser controls: R = reset session, S = save snapshot. Final report is locked until SET 1 three reps + SET 2 three reps are completed.</div>
        <div class="grid">
          <div class="item"><div class="label">Live Tracking Quality</div><div id="quality" class="value">-</div></div>
          <div class="item"><div class="label">Last Rep Quality</div><div id="lastRepQuality" class="value">-</div></div>
          <div class="item"><div class="label">Report Source</div><div id="reportSource" class="value">-</div></div>
          <div class="item"><div class="label">Camera Guide</div><div id="guide" class="value">-</div></div>
          <div class="item"><div class="label">Sensor Mode</div><div id="sensor" class="value">-</div></div>
          <div class="item"><div class="label">Phase</div><div id="phase" class="value">-</div></div>
          <div class="item"><div class="label">Confidence</div><div id="confidence" class="value">-</div></div>
          <div class="item"><div class="label">Errors</div><div id="errors" class="value error">-</div></div>
          <div class="item"><div class="label">Selected Feedback</div><div id="feedback" class="value feedback">-</div></div>
        </div>
      </section>
      <section style="margin-top:14px">
        <h2>Quality Evidence</h2>
        <pre id="qualityEvidence" class="evidence">Finish a rep to see collected evidence and decision reasons.</pre>
        <table style="margin-top:10px">
          <thead>
            <tr>
              <th>Phase</th>
              <th>Level</th>
              <th>Frames</th>
              <th>Capture</th>
              <th>Conf</th>
              <th>Lower Vis</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody id="phaseEvidence"></tbody>
        </table>
      </section>
      <section style="margin-top:14px">
        <h2>Metrics</h2>
        <pre id="metrics">-</pre>
      </section>
      <section style="margin-top:14px">
        <h2>AI Coach Report</h2>
        <pre id="report">Press Space/Enter or click Finish Rep / Report after a rep.</pre>
      </section>
    </div>
  </main>
  <script>
    async function sendCommand(command) {
      try {
        const res = await fetch('/command', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ command })
        });
        setText('commandStatus', `Command status: ${command} ${res.ok ? 'sent' : 'failed'}`);
      } catch (e) {
        setText('commandStatus', `Command status: ${command} failed (${e})`);
      }
    }
    window.addEventListener('keydown', (event) => {
      if (event.key === 'r' || event.key === 'R') {
        sendCommand('reset_rep');
      } else if (event.key === 's' || event.key === 'S') {
        sendCommand('save_snapshot');
      }
    });
    function levelColor(level) {
      return level === 'bad' ? '#ff8d8d' : (level === 'weak' ? '#ffe780' : '#8dffad');
    }
    function fmt(value) {
      if (value === undefined || value === null || Number.isNaN(Number(value))) return '-';
      return Number(value).toFixed(3);
    }
    function setText(id, text) {
      document.getElementById(id).textContent = text;
    }
    function renderEvidence(s) {
      const live = s.live_tracking_quality || s.tracking_quality || {};
      const last = s.last_rep_quality || {};
      const summary = s.latest_rep_summary || {};
      const rule = summary.rule_analysis || {};
      const metrics = summary.metrics || {};
      const lines = [
        `Live Reason: ${live.reason || '-'}`,
        `Live Missing Landmarks: ${(live.missing_landmarks || []).join(', ') || 'None'}`,
        `Live Recommendation: ${live.recommendation || '-'}`,
        '',
        `Last Rep Reason: ${last.reason || '-'}`,
        `Report Allowed: ${last.report_allowed === undefined ? '-' : (last.report_allowed ? 'YES' : 'NO')}`,
        `Mean Confidence: ${fmt(last.mean_confidence)}`,
        `Mean Lower Body Visibility: ${fmt(last.mean_lower_body_visibility)}`,
        `Sensor Mode: ${summary.sensor_mode || '-'}`,
        `Rule Main Issue: ${summary.main_issue || rule.main_issue || 'None'}`,
        `Detected Errors: ${(summary.detected_errors || rule.detected_errors || []).join(', ') || 'None'}`,
        `Selected Feedback: ${summary.selected_feedback || 'None'}`,
        `Key Metrics: min_knee_angle=${fmt(metrics.min_knee_angle)}, max_torso_angle=${fmt(metrics.max_torso_angle)}, max_knee_valgus_score=${fmt(metrics.max_knee_valgus_score)}`
      ];
      setText('qualityEvidence', lines.join('\n'));

      const body = document.getElementById('phaseEvidence');
      body.replaceChildren();
      const phases = ['standing', 'descent', 'bottom', 'ascent'];
      const phaseQuality = summary.phase_quality || last.phase_quality || {};
      for (const phase of phases) {
        const item = phaseQuality[phase] || {};
        const itemLevel = item.quality || item.level;
        const row = document.createElement('tr');
        const values = [
          phase,
          String(itemLevel || '-').toUpperCase(),
          item.frame_count ?? 0,
          item.has_capture ? 'YES' : 'NO',
          fmt(item.confidence ?? item.mean_confidence),
          fmt(item.lower_body_visibility ?? item.mean_lower_body_visibility),
          item.reason || '-'
        ];
        values.forEach((value, index) => {
          const cell = document.createElement('td');
          cell.textContent = value;
          if (index === 1) cell.style.color = levelColor(itemLevel);
          if (index === 6) cell.className = 'reason';
          row.appendChild(cell);
        });
        body.appendChild(row);
      }
    }
    async function tick() {
      try {
        const res = await fetch('/state.json', { cache: 'no-store' });
        const s = await res.json();
        const q = s.live_tracking_quality || s.tracking_quality || {};
        const level = q.quality_level || '-';
        const qualityEl = document.getElementById('quality');
        qualityEl.textContent = String(level).toUpperCase();
        qualityEl.style.color = levelColor(level);
        const last = s.last_rep_quality || {};
        const lastLevel = last.level || '-';
        const lastEl = document.getElementById('lastRepQuality');
        lastEl.textContent = String(lastLevel).toUpperCase();
        lastEl.style.color = levelColor(lastLevel);
        setText('reportSource', s.report_source || '-');
        setText('guide', level === 'bad' ? (q.recommendation || 'Move camera back until full body is visible.') : 'OK');
        const sensorSource = s.sensor?.source || '-';
        setText('sensor', sensorSource === 'mock' ? 'MOCK SENSOR' : String(sensorSource).toUpperCase());
        setText('phase', s.phase ?? '-');
        setText('confidence', Number(s.confidence ?? 0).toFixed(2));
        setText('errors', (s.detected_errors || []).join(', ') || 'None');
        setText('feedback', s.realtime_feedback || '-');
        setText('metrics', JSON.stringify(s.metrics || {}, null, 2));
        renderEvidence(s);
        if (s.latest_report) setText('report', s.latest_report);
      } catch (e) {
        setText('commandStatus', `State update failed: ${e}`);
      }
      setTimeout(tick, 500);
    }
    tick();
  </script>
</body>
</html>
"""
