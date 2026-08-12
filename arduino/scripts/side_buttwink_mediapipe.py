import argparse
import json
import math
import platform
import socket
import time
import urllib.request
from typing import Any

import cv2
import mediapipe as mp
import numpy as np


LANDMARK_NAMES = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    31: "left_foot",
    32: "right_foot",
}


class MjpegHttpCapture:
    def __init__(self, url: str):
        self.url = url
        self.buffer = b""
        self.response = None
        self.opened = False
        request = urllib.request.Request(url, headers={"User-Agent": "SafeLiftXR/1.0"})
        self.response = urllib.request.urlopen(request, timeout=10)
        self.opened = True

    def isOpened(self) -> bool:
        return self.opened

    def read(self):
        if not self.opened or self.response is None:
            return False, None

        deadline = time.time() + 5.0
        while time.time() < deadline:
            chunk = self.response.read(4096)
            if not chunk:
                self.release()
                return False, None

            self.buffer += chunk
            start = self.buffer.find(b"\xff\xd8")
            end = self.buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
            if start >= 0 and end >= 0:
                jpg = self.buffer[start : end + 2]
                self.buffer = self.buffer[end + 2 :]
                array = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
                if frame is not None and frame.size > 0:
                    return True, frame

        return False, None

    def release(self) -> None:
        self.opened = False
        if self.response is not None:
            try:
                self.response.close()
            except Exception:
                pass
            self.response = None


def camera_backends(preferred: str = "auto"):
    preferred = (preferred or "auto").upper()
    mapping = {
        "DSHOW": cv2.CAP_DSHOW,
        "MSMF": cv2.CAP_MSMF,
        "ANY": cv2.CAP_ANY,
    }
    if preferred in mapping:
        return [(preferred, mapping[preferred])]
    if platform.system().lower() == "windows":
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
    return [("ANY", cv2.CAP_ANY)]


def fourcc_candidates(preferred: str = "auto"):
    preferred = (preferred or "auto").upper()
    if preferred == "NONE":
        return [None]
    values = []
    if preferred not in ("AUTO", ""):
        values.append(preferred[:4])
    values.extend([None, "MJPG", "YUY2"])
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def size_candidates(width: int, height: int):
    values = [(width, height), (640, 480), (640, 360), (320, 240), (1280, 720)]
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def apply_capture_settings(cap, width: int, height: int, fourcc: str | None):
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def read_first_frame(cap, warmup_seconds: float):
    deadline = time.time() + max(0.5, warmup_seconds)
    last_frame = None
    while time.time() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True, frame

        grabbed = cap.grab()
        if grabbed:
            ok, frame = cap.retrieve()
            if ok and frame is not None and frame.size > 0:
                return True, frame
            last_frame = frame
        time.sleep(0.05)
    return False, last_frame


def open_video_source(source: str, warmup_seconds: float):
    print(f"Trying video source: {source}")
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        ok, frame = read_first_frame(cap, warmup_seconds)
        if ok and frame is not None and frame.size > 0:
            print(f"Selected video source: {source}, frame={frame.shape[1]}x{frame.shape[0]}")
            return cap

    cap.release()

    if source.startswith("http://") or source.startswith("https://"):
        print(f"Trying HTTP MJPEG reader: {source}")
        try:
            mjpeg_cap = MjpegHttpCapture(source)
            ok, frame = read_first_frame(mjpeg_cap, warmup_seconds)
            if ok and frame is not None and frame.size > 0:
                print(f"Selected HTTP MJPEG source: {source}, frame={frame.shape[1]}x{frame.shape[0]}")
                return mjpeg_cap
            mjpeg_cap.release()
        except Exception as exc:
            raise RuntimeError(f"HTTP MJPEG reader failed for {source}: {exc}") from exc

    raise RuntimeError(f"Could not open video source: {source}")


def droidcam_sources(ip: str, port: int):
    base = f"http://{ip}:{port}"
    return [
        f"{base}/video",
        f"{base}/video?640x480",
        f"{base}/video?320x240",
        f"{base}/mjpegfeed",
        f"{base}/mjpegfeed?640x480",
        f"{base}/?action=stream",
    ]


def open_droidcam(ip: str, port: int, warmup_seconds: float):
    errors = []
    for source in droidcam_sources(ip, port):
        try:
            return open_video_source(source, warmup_seconds)
        except RuntimeError as exc:
            errors.append(str(exc))

    print("DroidCam URL attempts failed:")
    for error in errors:
        print("  - " + error)
    raise RuntimeError(
        f"Could not open DroidCam at {ip}:{port}. "
        f"First check in a browser: http://{ip}:{port}/video"
    )


def try_open_camera(index: int, width: int, height: int, backend_preference: str, fourcc_preference: str, warmup_seconds: float):
    errors = []
    for backend_name, backend in camera_backends(backend_preference):
        for fourcc in fourcc_candidates(fourcc_preference):
            for requested_width, requested_height in size_candidates(width, height):
                print(
                    f"Trying camera {index}: backend={backend_name}, "
                    f"fourcc={fourcc or 'default'}, size={requested_width}x{requested_height}"
                )
                cap = cv2.VideoCapture(index, backend)
                if not cap.isOpened():
                    errors.append(f"{backend_name}/{fourcc or 'default'}/{requested_width}x{requested_height}: open failed")
                    cap.release()
                    continue

                apply_capture_settings(cap, requested_width, requested_height, fourcc)
                ok, frame = read_first_frame(cap, warmup_seconds)
                if ok and frame is not None and frame.size > 0:
                    print(
                        f"Selected side camera index={index}, backend={backend_name}, "
                        f"fourcc={fourcc or 'default'}, frame={frame.shape[1]}x{frame.shape[0]}"
                    )
                    return cap, []

                errors.append(f"{backend_name}/{fourcc or 'default'}/{requested_width}x{requested_height}: no frame")
                cap.release()

    return None, errors


def open_camera(index: int, width: int, height: int, backend_preference: str, fourcc_preference: str, warmup_seconds: float, fallback_scan: bool):
    cap, errors = try_open_camera(index, width, height, backend_preference, fourcc_preference, warmup_seconds)
    if cap is not None:
        return cap

    # In this setup the user wants:
    #   0 = laptop webcam, 1 = USB webcam, 2 = DroidCam.
    # However Windows/OpenCV sometimes exposes DroidCam at another DirectShow index
    # such as 6.  If index 2 cannot be opened, scan likely DroidCam candidates
    # while deliberately skipping 0 and 1 so we do not accidentally select the
    # laptop/USB cameras.
    if fallback_scan:
        print(f"Camera {index} did not open. Scanning DroidCam fallback candidates, skipping 0 and 1...")
        for fallback_index in [6, 3, 4, 5, 7, 8, 9, 10]:
            if fallback_index == index:
                continue
            cap, fallback_errors = try_open_camera(
                fallback_index,
                width,
                height,
                backend_preference,
                fourcc_preference,
                warmup_seconds,
            )
            if cap is not None:
                print(f"Using fallback camera index {fallback_index}. If this is DroidCam, keep using --camera {fallback_index}.")
                return cap
            errors.extend([f"index {fallback_index}: {error}" for error in fallback_errors[-4:]])

    print("Camera open attempts failed:")
    for error in errors[-20:]:
        print("  - " + error)
    raise RuntimeError(
        f"Could not open camera index {index}. "
        "Close Windows Camera, Unity Play mode, old Python MediaPipe windows, and any app using DroidCam. "
        "If DroidCam Client preview is open, close only the preview/client window and keep the phone connected."
    )


def landmark_dict(results) -> dict[int, dict[str, float]]:
    if results.pose_landmarks is None:
        return {}
    return {
        i: {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility),
        }
        for i, lm in enumerate(results.pose_landmarks.landmark)
    }


def point(landmarks: dict[int, dict[str, float]], index: int):
    lm = landmarks.get(index)
    if lm is None:
        return None
    return np.array([float(lm["x"]), float(lm["y"])], dtype=float)


def midpoint(landmarks: dict[int, dict[str, float]], a: int, b: int):
    pa = point(landmarks, a)
    pb = point(landmarks, b)
    if pa is None or pb is None:
        return None
    return (pa + pb) / 2.0


def angle_at(landmarks: dict[int, dict[str, float]], a: int, b: int, c: int):
    pa = point(landmarks, a)
    pb = point(landmarks, b)
    pc = point(landmarks, c)
    if pa is None or pb is None or pc is None:
        return None
    ba = pa - pb
    bc = pc - pb
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-8:
        return None
    cos_value = float(np.dot(ba, bc) / denom)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))


def torso_angle_from_vertical(landmarks: dict[int, dict[str, float]]):
    shoulder_mid = midpoint(landmarks, 11, 12)
    hip_mid = midpoint(landmarks, 23, 24)
    if shoulder_mid is None or hip_mid is None:
        return None
    vector = shoulder_mid - hip_mid
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return None
    vertical_up = np.array([0.0, -1.0], dtype=float)
    cos_value = float(np.dot(vector, vertical_up) / norm)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))


def avg_visibility(landmarks: dict[int, dict[str, float]], indexes: list[int]):
    values = [float(landmarks[i]["visibility"]) for i in indexes if i in landmarks]
    return float(sum(values) / len(values)) if values else 0.0


def compute_metrics(landmarks: dict[int, dict[str, float]], prev_hip_y, dt):
    left_knee = angle_at(landmarks, 23, 25, 27)
    right_knee = angle_at(landmarks, 24, 26, 28)
    knee_values = [v for v in [left_knee, right_knee] if v is not None]
    avg_knee = sum(knee_values) / len(knee_values) if knee_values else None
    hip_mid = midpoint(landmarks, 23, 24)
    hip_y = float(hip_mid[1]) if hip_mid is not None else None
    hip_velocity = None
    if hip_y is not None and prev_hip_y is not None and dt > 0:
        hip_velocity = (hip_y - prev_hip_y) / dt

    return {
        "torso_angle": torso_angle_from_vertical(landmarks),
        "left_knee_angle": left_knee,
        "right_knee_angle": right_knee,
        "avg_knee_angle": avg_knee,
        "hip_height": hip_y,
        "hip_velocity": hip_velocity,
    }


def phase_from_metrics(metrics: dict[str, Any]):
    knee = metrics.get("avg_knee_angle")
    velocity = metrics.get("hip_velocity")
    if knee is None:
        return "Unknown"
    if knee > 160:
        return "Standing"
    if knee < 120:
        return "Bottom"
    if velocity is None:
        return "Squat"
    return "Descent" if velocity > 0.02 else "Ascent"


def build_udp_packet(frame_id: int, landmarks: dict[int, dict[str, float]], confidence: float, phase: str, metrics: dict[str, Any], width: int, height: int):
    return {
        "schema": "gist-hai-avatar-pose-v1",
        "source": "side-mediapipe-standalone",
        "timestamp": time.time(),
        "frame_id": frame_id,
        "image": {"width": width, "height": height, "flipped": False},
        "tracking": {"pose_detected": bool(landmarks), "confidence": float(confidence)},
        "targets": [],
        "landmarks": [
            {
                "id": int(i),
                "name": LANDMARK_NAMES.get(i, str(i)),
                "x": float(lm["x"]),
                "y": float(lm["y"]),
                "z": float(lm["z"]),
                "visibility": float(lm["visibility"]),
                "presence": float(lm["visibility"]),
                "position": {
                    "x": float(lm["x"]) - 0.5,
                    "y": 1.0 - float(lm["y"]),
                    "z": -float(lm["z"]),
                },
            }
            for i, lm in landmarks.items()
        ],
        "analysis": {
            "phase": phase,
            "confidence": float(confidence),
            "left_knee_angle": float(metrics.get("left_knee_angle") or 0.0),
            "right_knee_angle": float(metrics.get("right_knee_angle") or 0.0),
            "avg_knee_angle": float(metrics.get("avg_knee_angle") or 0.0),
            "torso_angle_from_vertical": float(metrics.get("torso_angle") or 0.0),
            "errors": [],
            "tracking_uncertain": confidence < 0.10,
            "knee_valgus": False,
            "torso_lean": float(metrics.get("torso_angle") or 0.0) > 25.0,
            "shallow_depth": False,
            "asymmetry": False,
        },
    }


def draw_status(frame, metrics, phase, confidence, neutral_torso, torso_delta, warning):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 138), (0, 0, 0), -1)
    color = (0, 255, 0) if warning == "GOOD" else (0, 220, 255) if warning == "CAUTION" else (0, 0, 255)
    torso = metrics.get("torso_angle")
    knee = metrics.get("avg_knee_angle")
    cv2.putText(frame, f"Side MediaPipe Butt Wink Monitor", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(frame, f"conf={confidence:.2f} phase={phase}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    cv2.putText(frame, f"torso={torso or 0:.1f} neutral={neutral_torso or 0:.1f} delta={torso_delta:+.1f}", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
    cv2.putText(frame, f"knee={knee or 0:.1f} signal={warning}   press C=calibrate Q=quit", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone side-camera MediaPipe squat/butt-wink monitor.")
    parser.add_argument("--camera", type=int, default=2, help="Side camera index. Default 2 = DroidCam in this setup.")
    parser.add_argument("--source", default="", help="Direct video source URL/path. Example: http://PHONE_IP:4747/video")
    parser.add_argument("--droidcam-ip", default="", help="Phone IP address shown in the DroidCam app. Uses http://IP:4747/video")
    parser.add_argument("--droidcam-port", type=int, default=4747)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--backend", default="auto", choices=["auto", "DSHOW", "MSMF", "ANY"], help="OpenCV backend. DroidCam often works with DSHOW or MSMF.")
    parser.add_argument("--fourcc", default="auto", help="auto, NONE, MJPG, or YUY2")
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--no-fallback-scan", action="store_true", help="Only try the requested camera index.")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5005)
    parser.add_argument("--no-udp", action="store_true")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()

    if args.source:
        cap = open_video_source(args.source, args.warmup_seconds)
    elif args.droidcam_ip:
        cap = open_droidcam(args.droidcam_ip, args.droidcam_port, args.warmup_seconds)
    else:
        cap = open_camera(
            args.camera,
            args.width,
            args.height,
            args.backend,
            args.fourcc,
            args.warmup_seconds,
            not args.no_fallback_scan,
        )
    udp = None if args.no_udp else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_addr = (args.udp_host, args.udp_port)
    if udp:
        print(f"Sending Unity pose UDP to {args.udp_host}:{args.udp_port}")

    mp_pose = mp.solutions.pose
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose(
        model_complexity=args.model_complexity,
        enable_segmentation=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    neutral_torso = None
    prev_hip_y = None
    prev_time = time.time()
    frame_id = 0

    print("Controls: C = calibrate standing torso angle, Q/Esc = quit")
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera read failed.")
                break

            now = time.time()
            dt = now - prev_time
            prev_time = now
            frame_id += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            landmarks = landmark_dict(results)
            confidence = avg_visibility(landmarks, [11, 12, 23, 24, 25, 26, 27, 28]) if landmarks else 0.0
            metrics = compute_metrics(landmarks, prev_hip_y, dt) if landmarks else {}
            if metrics.get("hip_height") is not None:
                prev_hip_y = metrics["hip_height"]
            phase = phase_from_metrics(metrics)

            torso = metrics.get("torso_angle")
            if neutral_torso is None and torso is not None and confidence >= 0.35 and phase == "Standing":
                neutral_torso = torso

            torso_delta = 0.0 if neutral_torso is None or torso is None else torso - neutral_torso
            knee = metrics.get("avg_knee_angle")
            deep = knee is not None and knee <= 135.0
            warning = "GOOD"
            if deep and abs(torso_delta) >= 10.0:
                warning = "BUTT WINK?"
            elif deep and abs(torso_delta) >= 6.0:
                warning = "CAUTION"

            if results.pose_landmarks:
                drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                )

            draw_status(frame, metrics, phase, confidence, neutral_torso, torso_delta, warning)

            if udp:
                packet = build_udp_packet(frame_id, landmarks, confidence, phase, metrics, frame.shape[1], frame.shape[0])
                data = json.dumps(packet, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                udp.sendto(data, udp_addr)

            cv2.imshow("Standalone Side MediaPipe Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                if torso is not None:
                    neutral_torso = torso
                    print(f"Calibrated neutral torso angle: {neutral_torso:.2f}")
                else:
                    print("Cannot calibrate: torso landmarks not detected.")
    finally:
        pose.close()
        cap.release()
        if udp:
            udp.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
