import argparse
import math
import socket
import threading
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class ImuFrame:
    pitch: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    acc_x: float = 0.0
    acc_y: float = 0.0
    acc_z: float = 0.0
    quat_w: float = 1.0
    quat_x: float = 0.0
    quat_y: float = 0.0
    quat_z: float = 0.0
    angle_source: str = "none"
    timestamp: float = -1.0


class ImuUdpReader:
    def __init__(self, port: int):
        self.port = port
        self.latest = ImuFrame()
        self.last_line = ""
        self.last_parse_error = ""
        self.packet_count = 0
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def is_fresh(self, stale_after: float = 1.0) -> bool:
        return self.latest.timestamp > 0 and time.time() - self.latest.timestamp <= stale_after

    def age_seconds(self) -> float:
        if self.latest.timestamp <= 0:
            return float("inf")
        return time.time() - self.latest.timestamp

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", self.port))
        except OSError:
            sock.bind(("", self.port))
        sock.settimeout(0.2)

        while self.running:
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            line = data.decode("utf-8", errors="ignore").strip()
            self.last_line = line
            frame, error = parse_slxr_line(line)
            if frame is not None:
                self.latest = frame
                self.packet_count += 1
                self.last_parse_error = ""
            else:
                self.last_parse_error = error

        sock.close()


def parse_slxr_line(line: str) -> tuple[ImuFrame | None, str]:
    parts = line.split(",")
    if len(parts) < 18 or parts[0] != "SLXR":
        return None, "not SLXR or too short"

    try:
        acc_x = float(parts[8])
        acc_y = float(parts[9])
        acc_z = float(parts[10])
        qw = float(parts[14])
        qx = float(parts[15])
        qy = float(parts[16])
        qz = float(parts[17])
    except ValueError:
        return None, "accel/quaternion parse failed"

    quat_norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if quat_norm >= 0.5:
        qw, qx, qy, qz = (qw / quat_norm, qx / quat_norm, qy / quat_norm, qz / quat_norm)
        pitch, roll, yaw = quat_to_euler_degrees(qw, qx, qy, qz)
        angle_source = "quat"
    else:
        # Only fall back when the quaternion is invalid. Identity is a valid
        # physical orientation and must not cause source switching mid-motion.
        pitch, roll = accel_to_tilt_degrees(acc_x, acc_y, acc_z)
        yaw = 0.0
        angle_source = "accel"

    return ImuFrame(
        pitch=pitch,
        roll=roll,
        yaw=yaw,
        acc_x=acc_x,
        acc_y=acc_y,
        acc_z=acc_z,
        quat_w=qw,
        quat_x=qx,
        quat_y=qy,
        quat_z=qz,
        angle_source=angle_source,
        timestamp=time.time(),
    ), ""


def accel_to_tilt_degrees(ax: float, ay: float, az: float):
    # These are relative tilt estimates from gravity.  Absolute sign depends on
    # how the IMU is mounted, but calibration makes the delta useful.
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    roll = math.degrees(math.atan2(ay, az))
    return normalize_angle(pitch), normalize_angle(roll)


def quat_to_euler_degrees(w: float, x: float, y: float, z: float):
    # Same practical convention as Unity's Quaternion.eulerAngles.x/y/z:
    # pitch is rotation around X, yaw around Y, roll around Z, normalized to -180..180.
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    pitch_x = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        yaw_y = math.degrees(math.copysign(math.pi / 2.0, sinp))
    else:
        yaw_y = math.degrees(math.asin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    roll_z = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return normalize_angle(pitch_x), normalize_angle(roll_z), normalize_angle(yaw_y)


def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def delta_angle(a: float, b: float) -> float:
    return normalize_angle(b - a)


def pixel_point(landmark, width: int, height: int):
    if landmark is None:
        return None
    return np.array([landmark.x * width, landmark.y * height], dtype=float)


def signed_angle_from_vertical(hip, shoulder, width: int, height: int) -> float | None:
    hip_point = pixel_point(hip, width, height)
    shoulder_point = pixel_point(shoulder, width, height)
    if hip_point is None or shoulder_point is None:
        return None
    vector = shoulder_point - hip_point
    if float(np.linalg.norm(vector)) <= 1e-8:
        return None
    # atan2 preserves the camera-side direction.  Zero is upright; positive
    # means the shoulder is to the right of the hip in the displayed image.
    return math.degrees(math.atan2(float(vector[0]), float(-vector[1])))


def joint_angle(a, b, c, width: int, height: int) -> float | None:
    if a is None or b is None or c is None:
        return None
    pa = pixel_point(a, width, height)
    pb = pixel_point(b, width, height)
    pc = pixel_point(c, width, height)
    ba = pa - pb
    bc = pc - pb
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom <= 1e-8:
        return None
    cos_value = float(np.dot(ba, bc) / denom)
    cos_value = max(-1.0, min(1.0, cos_value))
    return math.degrees(math.acos(cos_value))


def pose_metrics(results, width: int, height: int):
    if results.pose_landmarks is None:
        return None

    lm = results.pose_landmarks.landmark
    side_ids = {
        "left": (11, 23, 25, 27),
        "right": (12, 24, 26, 28),
    }
    side_confidence = {
        side: sum(float(lm[i].visibility) for i in ids) / len(ids)
        for side, ids in side_ids.items()
    }
    side = max(side_confidence, key=side_confidence.get)
    shoulder_id, hip_id, knee_id, ankle_id = side_ids[side]
    confidence = side_confidence[side]
    torso = signed_angle_from_vertical(lm[hip_id], lm[shoulder_id], width, height)
    knee = joint_angle(lm[hip_id], lm[knee_id], lm[ankle_id], width, height)

    return {
        "confidence": confidence,
        "torso_angle": torso,
        "avg_knee_angle": knee,
        "tracking_side": side,
    }


def frame_quaternion(frame: ImuFrame):
    values = np.array([frame.quat_w, frame.quat_x, frame.quat_y, frame.quat_z], dtype=float)
    norm = float(np.linalg.norm(values))
    if frame.angle_source != "quat" or norm < 0.5:
        return None
    return values / norm


def average_quaternions(samples):
    if not samples:
        return None
    reference = samples[0]
    aligned = [sample if float(np.dot(sample, reference)) >= 0.0 else -sample for sample in samples]
    average = np.sum(aligned, axis=0)
    norm = float(np.linalg.norm(average))
    return average / norm if norm > 1e-8 else None


def quaternion_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def relative_pitch_twist_degrees(neutral, current) -> float:
    # q_rel is expressed in the calibrated sensor frame.  Swing-twist
    # decomposition then isolates rotation about the sensor's pitch/X axis,
    # avoiding Euler pitch contamination from its large roll/yaw orientation.
    neutral_inverse = np.array([neutral[0], -neutral[1], -neutral[2], -neutral[3]], dtype=float)
    relative = quaternion_multiply(neutral_inverse, current)
    if relative[0] < 0.0:
        relative = -relative
    twist_norm = math.hypot(float(relative[0]), float(relative[1]))
    if twist_norm <= 1e-8:
        return 0.0
    twist_w = float(relative[0]) / twist_norm
    twist_x = float(relative[1]) / twist_norm
    return normalize_angle(math.degrees(2.0 * math.atan2(twist_x, twist_w)))


def pelvis_pitch_value(frame: ImuFrame) -> float:
    # Sensor/body mapping confirmed for this setup:
    #   pitch = anterior/posterior pelvic rotation (butt-wink signal)
    #   roll  = rotation around the spinal axis
    #   yaw   = left/right body rotation
    # Only pitch is allowed to drive pelvic-tuck detection.
    return frame.pitch


def ema_angle(previous: float | None, current: float, alpha: float) -> float:
    if previous is None:
        return current
    return normalize_angle(previous + max(0.0, min(1.0, alpha)) * delta_angle(previous, current))


def ema_value(previous: float | None, current: float, alpha: float) -> float:
    if previous is None:
        return current
    return previous + max(0.0, min(1.0, alpha)) * (current - previous)


def open_camera(index: int, width: int, height: int, label: str):
    for backend_name, backend in [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(30):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(
                    f"Selected {label} camera {index} "
                    f"backend={backend_name} frame={frame.shape[1]}x{frame.shape[0]}"
                )
                return cap
            time.sleep(0.03)
        cap.release()
    raise RuntimeError(f"Could not open {label} camera index {index}")


def open_droidcam_stream(ip: str, port: int, label: str):
    urls = [
        f"http://{ip}:{port}/video",
        f"http://{ip}:{port}/mjpegfeed",
    ]
    errors = []
    for url in urls:
        print(f"Trying {label} DroidCam stream: {url}")
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            errors.append(f"{url}: could not open")
            cap.release()
            continue

        for _ in range(60):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(
                    f"Selected {label} DroidCam stream {url} "
                    f"frame={frame.shape[1]}x{frame.shape[0]}"
                )
                return cap
            time.sleep(0.05)
        errors.append(f"{url}: opened but returned no frames")
        cap.release()

    raise RuntimeError(
        f"Could not open {label} DroidCam stream at {ip}:{port}. "
        f"Open http://{ip}:{port}/video in a browser first. "
        f"Attempts: {'; '.join(errors)}"
    )


def crop_black_borders(frame, threshold: int = 8, min_active_fraction: float = 0.02):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    active = gray > threshold
    active_rows = np.mean(active, axis=1) >= min_active_fraction
    active_cols = np.mean(active, axis=0) >= min_active_fraction
    row_ids = np.flatnonzero(active_rows)
    col_ids = np.flatnonzero(active_cols)
    if row_ids.size == 0 or col_ids.size == 0:
        return frame
    return frame[row_ids[0]:row_ids[-1] + 1, col_ids[0]:col_ids[-1] + 1]


def crop_to_aspect(frame, target_width: int, target_height: int, vertical_align: str = "top"):
    height, width = frame.shape[:2]
    target_aspect = target_width / target_height
    current_aspect = width / height

    if current_aspect < target_aspect:
        crop_height = max(1, min(height, round(width / target_aspect)))
        y0 = 0 if vertical_align == "top" else max(0, (height - crop_height) // 2)
        return frame[y0:y0 + crop_height, :]

    if current_aspect > target_aspect:
        crop_width = max(1, min(width, round(height * target_aspect)))
        x0 = max(0, (width - crop_width) // 2)
        return frame[:, x0:x0 + crop_width]

    return frame


def make_dashboard(frame, lines, status_color):
    panel_width = 720
    source_h, source_w = frame.shape[:2]
    display_h = max(source_h, 720)
    display_w = max(1, round(source_w * display_h / source_h))
    display_frame = cv2.resize(frame, (display_w, display_h))
    h, w = display_frame.shape[:2]
    dashboard = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    dashboard[:h, :w] = display_frame
    dashboard[:, w:] = (18, 18, 18)
    cv2.rectangle(dashboard, (4, 4), (w - 5, h - 5), status_color, 6)
    cv2.rectangle(dashboard, (12, 12), (270, 48), (0, 0, 0), -1)
    cv2.putText(
        dashboard,
        "SIDE + IMU DETECTION",
        (22, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.line(dashboard, (w, 0), (w, h), (70, 70, 70), 2)
    cv2.putText(
        dashboard,
        "SafeLift Butt Wink",
        (w + 18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.rectangle(dashboard, (w + 14, 56), (w + panel_width - 14, 122), status_color, -1)
    cv2.rectangle(dashboard, (w + 14, 56), (w + panel_width - 14, 122), (255, 255, 255), 2)
    verdict = lines[0] if lines else "WAITING"
    cv2.putText(
        dashboard,
        verdict,
        (w + 28, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    wrapped_groups = [wrap_text(line, max_chars=62) for line in lines[1:]]
    row_count = max(1, sum(len(group) for group in wrapped_groups))
    group_gap = 2
    available_height = max(1, h - 202)
    line_height = max(
        15,
        min(25, (available_height - group_gap * len(wrapped_groups)) // row_count),
    )
    text_size = 0.52 if line_height >= 22 else 0.44
    y = 154
    for parts in wrapped_groups:
        color = (230, 230, 230)
        thickness = 1
        for part in parts:
            cv2.putText(
                dashboard,
                part,
                (w + 18, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_size,
                color,
                thickness,
                cv2.LINE_AA,
            )
            y += line_height
        y += group_gap

    cv2.putText(
        dashboard,
        "C: recalibrate    Q/Esc: quit",
        (w + 18, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (160, 210, 255),
        1,
        cv2.LINE_AA,
    )
    return dashboard


def calibration_progress(start_time: float, duration: float) -> tuple[float, float]:
    elapsed = max(0.0, time.time() - start_time)
    progress = min(1.0, elapsed / max(0.01, duration))
    remaining = max(0.0, duration - elapsed)
    return progress, remaining


def wrap_text(text: str, max_chars: int):
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) > max_chars:
            lines.append(current)
            current = word
        else:
            current += " " + word
    lines.append(current)
    return lines


def shorten(text: str, max_chars: int):
    if not text:
        return "-"
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def verdict_reason(
    has_imu: bool,
    has_pose: bool,
    squat_active: bool,
    standing_tuck: bool,
    butt_wink: bool,
):
    if not has_imu:
        return "IMU data is missing, so pelvis tilt cannot be judged."
    if not has_pose:
        return "MediaPipe pose is missing, so torso/knee angle cannot be judged."
    if squat_active:
        if butt_wink:
            return "Posterior pelvis-vs-torso rotation is sustained during the squat."
        return "Squat pelvis-to-torso relationship is within the normal range."
    if standing_tuck:
        return "Posterior pelvis-vs-torso rotation is sustained while standing."
    return "Standing pelvis-to-torso relationship is near the calibrated neutral."


def main() -> int:
    parser = argparse.ArgumentParser(description="Python-only butt wink monitor: side MediaPipe + pelvis IMU.")
    parser.add_argument(
        "--side-camera",
        "--camera",
        dest="side_camera",
        type=int,
        default=5,
        help="Side DroidCam OBS virtual-camera index. --camera remains as a compatible alias.",
    )
    parser.add_argument(
        "--front-camera",
        type=int,
        default=None,
        help="Optional front DroidCam OBS virtual-camera index. It is previewed only; detection still uses the side camera.",
    )
    parser.add_argument(
        "--obs-camera",
        type=int,
        default=None,
        help="OBS Virtual Camera index carrying a side-by-side front/side canvas.",
    )
    parser.add_argument(
        "--side-half",
        choices=["left", "right"],
        default="left",
        help="Which half of the OBS side-by-side canvas contains the side view.",
    )
    parser.add_argument(
        "--side-ip",
        default="",
        help="Side phone IP shown by DroidCam. When set, this overrides --side-camera.",
    )
    parser.add_argument("--side-port", type=int, default=4747)
    parser.add_argument(
        "--front-ip",
        default="",
        help="Front phone IP shown by DroidCam. When set, this overrides --front-camera.",
    )
    parser.add_argument("--front-port", type=int, default=4747)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--imu-udp-port", type=int, default=24017)
    parser.add_argument("--calibration-seconds", type=float, default=1.5)
    parser.add_argument("--pose-confidence", type=float, default=0.35)
    parser.add_argument("--knee-depth", type=float, default=120.0)
    parser.add_argument("--pelvis-sign", type=int, choices=[-1, 1], default=1,
                        help="Flip IMU pitch so forward pelvic rotation has the expected sign.")
    parser.add_argument("--torso-sign", type=int, choices=[-1, 1], default=1,
                        help="Flip the signed side-camera torso angle when the camera is mirrored.")
    parser.add_argument("--posterior-sign", type=int, choices=[-1, 1], default=1,
                        help="Direction of posterior pelvic rotation in the relative-angle signal.")
    parser.add_argument(
        "--standing-posterior-sign",
        type=int,
        choices=[-1, 1],
        default=-1,
        help="Posterior direction used only for calibrated standing pelvic-tuck detection.",
    )
    parser.add_argument(
        "--standing-imu-sign",
        type=int,
        choices=[-1, 1],
        default=1,
        help="Posterior direction of the pelvis IMU while standing.",
    )
    parser.add_argument("--baseline-knee", type=float, default=155.0,
                        help="Capture the per-rep relative-angle baseline before reaching squat depth.")
    parser.add_argument("--pelvis-warning", type=float, default=8.0,
                        help="Posterior direction-change angle required for BUTT WINK.")
    parser.add_argument("--mismatch-warning", type=float, default=5.0,
                        help="Posterior direction-change angle required for CAUTION.")
    parser.add_argument("--posterior-rate-warning", type=float, default=10.0,
                        help="Peak posterior relative-angle velocity required for BUTT WINK (deg/s).")
    parser.add_argument(
        "--bottom-imu-warning",
        type=float,
        default=4.0,
        help="Minimum IMU pelvis posterior reversal confirming bottom-phase Butt Wink.",
    )
    parser.add_argument(
        "--butt-wink-on-angle",
        "--descent-tuck-angle",
        dest="butt_wink_on_angle",
        type=float,
        default=6.0,
        help="Posterior relative-angle ON threshold for gradual squat butt wink.",
    )
    parser.add_argument(
        "--butt-wink-on-seconds",
        "--descent-tuck-hold-seconds",
        dest="butt_wink_on_seconds",
        type=float,
        default=0.15,
        help="How long gradual squat butt wink must exceed the ON threshold.",
    )
    parser.add_argument(
        "--descent-knee-velocity",
        type=float,
        default=3.0,
        help="Minimum knee-flexion speed used to enter the descending phase (deg/s).",
    )
    parser.add_argument("--filter-alpha", type=float, default=0.35,
                        help="EMA smoothing factor for IMU and MediaPipe angles (0..1).")
    parser.add_argument("--hold-seconds", type=float, default=0.12,
                        help="How long the warning threshold must be sustained.")
    parser.add_argument(
        "--max-ascent-knee-velocity",
        type=float,
        default=12.0,
        help="Maximum positive knee extension speed still treated as bottom-phase motion (deg/s).",
    )
    parser.add_argument(
        "--max-upright-torso-velocity",
        type=float,
        default=10.0,
        help="Maximum torso-straightening speed allowed during butt-wink detection (deg/s).",
    )
    parser.add_argument("--recovery-angle", type=float, default=3.0,
                        help="Butt-wink OFF threshold used for hysteresis recovery.")
    parser.add_argument("--recovery-seconds", type=float, default=0.40,
                        help="How long neutral recovery must be sustained before clearing BUTT WINK.")
    parser.add_argument("--standing-warning", type=float, default=8.0,
                        help="Posterior relative-angle change required for STANDING PELVIC TUCK.")
    parser.add_argument(
        "--standing-imu-warning",
        type=float,
        default=4.0,
        help="Minimum IMU pelvis posterior rotation confirming standing pelvic tuck.",
    )
    parser.add_argument("--standing-caution", type=float, default=6.5,
                        help="Posterior relative-angle change required for standing CAUTION.")
    parser.add_argument("--standing-knee", type=float, default=160.0,
                        help="Minimum knee angle treated as standing.")
    parser.add_argument("--standing-max-torso", type=float, default=30.0,
                        help="Maximum torso change allowed while standing tuck is evaluated.")
    parser.add_argument(
        "--standing-tuck-grace-seconds",
        type=float,
        default=1.0,
        help="How long a standing tuck is remembered while the squat starts.",
    )
    args = parser.parse_args()

    if (
        not args.side_ip
        and not args.front_ip
        and args.obs_camera is None
        and args.front_camera is not None
        and args.front_camera == args.side_camera
    ):
        parser.error("--front-camera and --side-camera must use different camera indexes.")
    if args.side_ip and args.front_ip and args.side_ip == args.front_ip and args.side_port == args.front_port:
        parser.error("--front-ip and --side-ip must identify different DroidCam streams.")
    if args.obs_camera is not None and (
        args.side_ip or args.front_ip or args.front_camera is not None
    ):
        parser.error(
            "--obs-camera already contains both views; do not combine it with "
            "--side-ip, --front-ip, or --front-camera."
        )

    imu = ImuUdpReader(args.imu_udp_port)
    if args.obs_camera is not None:
        side_cap = open_camera(args.obs_camera, args.width * 2, args.height, "OBS composite")
    elif args.side_ip:
        side_cap = open_droidcam_stream(args.side_ip, args.side_port, "side")
    else:
        side_cap = open_camera(args.side_camera, args.width, args.height, "side")
    try:
        if args.obs_camera is not None:
            front_cap = None
        elif args.front_ip:
            front_cap = open_droidcam_stream(args.front_ip, args.front_port, "front")
        elif args.front_camera is not None:
            front_cap = open_camera(args.front_camera, args.width, args.height, "front")
        else:
            front_cap = None
    except Exception:
        side_cap.release()
        raise
    imu.start()
    mp_pose = mp.solutions.pose
    drawing = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    front_pose = mp_pose.Pose(
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    neutral_pelvis = None
    neutral_quaternion = None
    neutral_torso = None
    quaternion_samples = []
    pelvis_sum = 0.0
    pelvis_count = 0
    torso_sum = 0.0
    torso_count = 0
    calibration_start = time.time()
    calibration_collecting = False
    filtered_pelvis = None
    filtered_torso = None
    filtered_knee = None
    previous_relative = None
    previous_knee = None
    previous_torso = None
    previous_motion_time = None
    filtered_posterior_velocity = 0.0
    filtered_torso_velocity = 0.0
    rep_active = False
    anterior_reference = None
    descending_phase = False
    descent_reference = None
    pelvis_anterior_reference = None
    pelvic_tuck_angle = 0.0
    pelvis_posterior_change = 0.0
    gradual_butt_wink_started_at = None
    max_posterior_velocity = 0.0
    butt_wink_latched = False
    recovery_started_at = None
    warning_started_at = None
    caution_started_at = None
    standing_warning_started_at = None
    standing_caution_started_at = None
    standing_recovery_started_at = None
    standing_tuck_latched = False
    last_standing_tuck_at = None
    pre_tucked_rep = False

    print("Python Butt Wink Monitor started.")
    print("Run Arduino bridge first so SLXR packets arrive on UDP 24017.")
    print("Controls: C = recalibrate, Q/Esc = quit")

    try:
        while True:
            ok, frame = side_cap.read()
            if not ok or frame is None:
                print("OBS/side camera read failed.")
                break
            front_frame = None
            if args.obs_camera is not None:
                split_x = frame.shape[1] // 2
                if split_x <= 0:
                    print("OBS composite frame is too narrow to split.")
                    break
                left_frame = frame[:, :split_x]
                right_frame = frame[:, split_x:]
                if args.side_half == "left":
                    frame, front_frame = left_frame, right_frame
                else:
                    frame, front_frame = right_frame, left_frame
                frame = crop_black_borders(frame)
                front_frame = crop_black_borders(front_frame)
                frame = crop_to_aspect(frame, args.width, args.height, vertical_align="top")
                front_frame = crop_to_aspect(
                    front_frame,
                    args.width,
                    args.height,
                    vertical_align="top",
                )
                frame = cv2.resize(frame, (args.width, args.height))
                front_frame = cv2.resize(front_frame, (args.width, args.height))
            elif front_cap is not None:
                front_ok, front_frame = front_cap.read()
                if not front_ok or front_frame is None:
                    print("Front camera read failed.")
                    break

            front_results = None
            if front_frame is not None:
                front_rgb = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
                front_results = front_pose.process(front_rgb)
                if front_results.pose_landmarks:
                    drawing.draw_landmarks(
                        front_frame,
                        front_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                    )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            metrics = pose_metrics(results, frame.shape[1], frame.shape[0])
            has_pose = metrics is not None and metrics["confidence"] >= args.pose_confidence and metrics["torso_angle"] is not None
            has_imu = imu.is_fresh()
            current_quaternion = frame_quaternion(imu.latest) if has_imu else None
            has_orientation = current_quaternion is not None
            imu_age = imu.age_seconds()
            imu_age_text = "never" if math.isinf(imu_age) else f"{imu_age:.2f}s ago"
            if has_pose:
                filtered_torso = ema_value(filtered_torso, metrics["torso_angle"], args.filter_alpha)
                if metrics["avg_knee_angle"] is not None:
                    filtered_knee = ema_value(filtered_knee, metrics["avg_knee_angle"], args.filter_alpha)
            if neutral_quaternion is not None and has_orientation:
                relative_pitch = relative_pitch_twist_degrees(neutral_quaternion, current_quaternion)
                filtered_pelvis = ema_angle(filtered_pelvis, relative_pitch, args.filter_alpha)

            if results.pose_landmarks:
                drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
                )

            if neutral_quaternion is None:
                calibration_upright = (
                    has_orientation
                    and has_pose
                    and metrics["avg_knee_angle"] is not None
                    and filtered_knee is not None
                    and filtered_knee >= args.standing_knee
                )
                if calibration_upright:
                    if not calibration_collecting:
                        calibration_collecting = True
                        calibration_start = time.time()
                        quaternion_samples = []
                        pelvis_sum = 0.0
                        pelvis_count = 0
                        torso_sum = 0.0
                        torso_count = 0
                        # Do not let a previously bent frame bias the neutral
                        # torso EMA when upright calibration begins.
                        filtered_torso = metrics["torso_angle"]
                        filtered_knee = metrics["avg_knee_angle"]
                    quaternion_samples.append(current_quaternion)
                    pelvis_sum += pelvis_pitch_value(imu.latest)
                    pelvis_count += 1
                    torso_sum += filtered_torso
                    torso_count += 1
                else:
                    calibration_collecting = False
                    calibration_start = time.time()
                    quaternion_samples = []
                    pelvis_sum = 0.0
                    pelvis_count = 0
                    torso_sum = 0.0
                    torso_count = 0

                progress, remaining = calibration_progress(
                    calibration_start,
                    args.calibration_seconds,
                )

                if (
                    time.time() - calibration_start >= args.calibration_seconds
                    and pelvis_count > 0
                    and torso_count > 0
                ):
                    neutral_pelvis = pelvis_sum / pelvis_count
                    neutral_quaternion = average_quaternions(quaternion_samples)
                    neutral_torso = torso_sum / torso_count

                lines = [
                    f"CALIBRATING {progress * 100:.0f}%",
                    f"Stand upright. Remaining {remaining:.1f}s",
                    f"Upright calibration hold={'OK' if calibration_upright else 'WAIT'} | knee {filtered_knee if filtered_knee is not None else 0:.1f}/{args.standing_knee:.1f}",
                    f"IMU orientation={'OK' if has_orientation else 'WAIT'} | Pose={'OK' if has_pose else 'WAIT'}",
                    f"Detection: relative quaternion PITCH | pelvis {args.pelvis_sign:+d} | squat posterior {args.posterior_sign:+d} | standing relative {args.standing_posterior_sign:+d} | standing IMU {args.standing_imu_sign:+d}",
                    f"IMU live ({imu.latest.angle_source}): pitch {imu.latest.pitch:+.1f} | roll {imu.latest.roll:+.1f} | yaw {imu.latest.yaw:+.1f} deg",
                    f"ACC: {imu.latest.acc_x:+.2f}, {imu.latest.acc_y:+.2f}, {imu.latest.acc_z:+.2f}",
                    f"QUAT: {imu.latest.quat_w:+.2f}, {imu.latest.quat_x:+.2f}, {imu.latest.quat_y:+.2f}, {imu.latest.quat_z:+.2f}",
                    f"IMU UDP: packets {imu.packet_count} | last {imu_age_text} | port {args.imu_udp_port}",
                    f"IMU raw: {shorten(imu.last_line, 48)}",
                    f"IMU parse: {imu.last_parse_error or 'OK'}",
                    f"pelvis samples={pelvis_count}, torso samples={torso_count}",
                    "Press C anytime to recalibrate.",
                ]
                display_frame = make_dashboard(frame, lines, (0, 220, 255))
            else:
                pelvis_delta = filtered_pelvis if has_orientation and filtered_pelvis is not None else 0.0
                torso_delta = (
                    filtered_torso - neutral_torso
                    if has_pose and filtered_torso is not None and neutral_torso is not None
                    else 0.0
                )
                knee = filtered_knee if has_pose and filtered_knee is not None else 0.0
                deep = has_pose and (knee <= args.knee_depth if knee > 0 else False)
                standing = has_pose and knee >= args.standing_knee
                relative_angle = (
                    args.pelvis_sign * pelvis_delta
                    - args.torso_sign * torso_delta
                )
                posterior_relative = args.posterior_sign * relative_angle
                posterior_pelvis = (
                    args.posterior_sign * args.pelvis_sign * pelvis_delta
                )
                now = time.time()
                knee_velocity = 0.0
                torso_velocity = 0.0
                if (
                    previous_motion_time is not None
                    and previous_relative is not None
                    and previous_knee is not None
                    and previous_torso is not None
                    and has_orientation
                    and has_pose
                ):
                    dt = max(1e-3, min(0.2, now - previous_motion_time))
                    raw_posterior_velocity = delta_angle(previous_relative, posterior_relative) / dt
                    filtered_posterior_velocity = ema_value(
                        filtered_posterior_velocity,
                        raw_posterior_velocity,
                        args.filter_alpha,
                    )
                    knee_velocity = (knee - previous_knee) / dt
                    raw_torso_velocity = (filtered_torso - previous_torso) / dt
                    filtered_torso_velocity = ema_value(
                        filtered_torso_velocity,
                        raw_torso_velocity,
                        args.filter_alpha,
                    )
                    torso_velocity = filtered_torso_velocity
                if has_orientation and has_pose:
                    previous_motion_time = now
                    previous_relative = posterior_relative
                    previous_knee = knee
                    previous_torso = filtered_torso

                # A normal ascent also reverses the calibrated pelvis-to-torso
                # angle, but it is not a butt wink. Only allow new warnings
                # while the knees are not extending quickly and the side-view
                # torso is not rapidly straightening.
                bottom_phase = (
                    knee_velocity <= args.max_ascent_knee_velocity
                    and torso_velocity >= -args.max_upright_torso_velocity
                )

                # MediaPipe knee motion determines the squat direction. Once
                # descent begins, keep the phase active through slow movement
                # and the bottom transition; clear it only when ascent is
                # clearly underway. This prevents a slow tuck from being lost
                # just because its per-frame knee velocity is near zero.
                if standing:
                    descending_phase = False
                elif has_pose:
                    if knee_velocity <= -args.descent_knee_velocity:
                        descending_phase = True
                    elif knee_velocity >= args.max_ascent_knee_velocity:
                        descending_phase = False

                # Each repetition supplies its own anterior reference. Normal
                # forward lean may accumulate with depth; a butt wink is the
                # later reversal away from the most-anterior relationship.
                if standing:
                    rep_active = False
                    pre_tucked_rep = False
                    anterior_reference = None
                    descent_reference = None
                    pelvis_anterior_reference = None
                    pelvic_tuck_angle = 0.0
                    pelvis_posterior_change = 0.0
                    gradual_butt_wink_started_at = None
                    max_posterior_velocity = 0.0
                    butt_wink_latched = False
                    recovery_started_at = None
                    warning_started_at = None
                    caution_started_at = None
                elif has_orientation and has_pose and 0.0 < knee <= args.baseline_knee:
                    if not rep_active:
                        rep_active = True
                        descending_phase = True
                        recent_standing_tuck = (
                            last_standing_tuck_at is not None
                            and now - last_standing_tuck_at
                            <= args.standing_tuck_grace_seconds
                        )
                        pre_tucked_rep = (
                            standing_tuck_latched or recent_standing_tuck
                        )
                        anterior_reference = posterior_relative
                        descent_reference = posterior_relative
                        pelvis_anterior_reference = posterior_pelvis
                        pelvic_tuck_angle = 0.0
                        pelvis_posterior_change = 0.0
                        gradual_butt_wink_started_at = None
                        max_posterior_velocity = (
                            max(0.0, filtered_posterior_velocity)
                            if bottom_phase
                            else 0.0
                        )
                        if pre_tucked_rep:
                            butt_wink_latched = True
                    else:
                        anterior_reference = min(anterior_reference, posterior_relative)
                        if descending_phase and descent_reference is not None:
                            descent_reference = min(descent_reference, posterior_relative)
                        if (
                            descending_phase
                            and pelvis_anterior_reference is not None
                        ):
                            pelvis_anterior_reference = min(
                                pelvis_anterior_reference,
                                posterior_pelvis,
                            )
                        if bottom_phase:
                            max_posterior_velocity = max(
                                max_posterior_velocity,
                                filtered_posterior_velocity,
                            )

                posterior_change = (
                    max(0.0, posterior_relative - anterior_reference)
                    if anterior_reference is not None
                    else 0.0
                )
                pelvic_tuck_angle = (
                    max(0.0, posterior_relative - descent_reference)
                    if descent_reference is not None
                    else 0.0
                )
                pelvis_posterior_change = (
                    max(0.0, posterior_pelvis - pelvis_anterior_reference)
                    if pelvis_anterior_reference is not None
                    else 0.0
                )
                gradual_butt_wink_candidate = (
                    has_orientation
                    and has_pose
                    and rep_active
                    and descending_phase
                    and not pre_tucked_rep
                    and descent_reference is not None
                    and pelvic_tuck_angle >= args.butt_wink_on_angle
                    # Normal squat torso lean can raise the relative angle by
                    # itself. Require the pelvis IMU to show a real posterior
                    # reversal as well before enabling gradual Butt Wink.
                    and pelvis_posterior_change >= args.pelvis_warning
                )
                if gradual_butt_wink_candidate:
                    if gradual_butt_wink_started_at is None:
                        gradual_butt_wink_started_at = now
                    elif (
                        now - gradual_butt_wink_started_at
                        >= args.butt_wink_on_seconds
                    ):
                        # Gradual descent tuck and the existing bottom-phase
                        # detector feed the same user-facing Butt Wink state.
                        butt_wink_latched = True
                else:
                    gradual_butt_wink_started_at = None

                warning_candidate = (
                    has_orientation and has_pose and deep and anterior_reference is not None
                    and bottom_phase
                    and posterior_change >= args.pelvis_warning
                    and max_posterior_velocity >= args.posterior_rate_warning
                    and pelvis_posterior_change >= args.bottom_imu_warning
                )
                caution_candidate = (
                    has_orientation and has_pose and deep and anterior_reference is not None
                    and bottom_phase
                    and posterior_change >= args.mismatch_warning
                )
                warning_started_at = now if warning_candidate and warning_started_at is None else warning_started_at
                caution_started_at = now if caution_candidate and caution_started_at is None else caution_started_at
                if not warning_candidate:
                    warning_started_at = None
                if not caution_candidate:
                    caution_started_at = None
                if warning_started_at is not None and now - warning_started_at >= args.hold_seconds:
                    butt_wink_latched = True

                # Real-time coaching recovery: once the pelvis-to-torso
                # relationship returns close to the anterior reference and
                # stays there, clear the previous warning without requiring
                # the user to stand up first.
                squat_recovery_angle = pelvis_posterior_change
                recovery_candidate = (
                    butt_wink_latched
                    and has_orientation
                    and has_pose
                    and rep_active
                    # If the repetition began from a confirmed standing
                    # pelvic tuck, keep Butt Wink for that repetition. Squat
                    # forward lean makes standing-neutral angles invalid for
                    # deciding recovery; standing mode will reassess it.
                    and not pre_tucked_rep
                    and squat_recovery_angle <= args.recovery_angle
                    # ON and OFF must never be true together. The previous
                    # version alternated every hold interval when the bottom
                    # detector and recovery detector disagreed.
                    and not warning_candidate
                    and not gradual_butt_wink_candidate
                )
                if recovery_candidate:
                    if recovery_started_at is None:
                        recovery_started_at = now
                    elif now - recovery_started_at >= args.recovery_seconds:
                        butt_wink_latched = False
                        warning_started_at = None
                        caution_started_at = None
                        recovery_started_at = None
                        pre_tucked_rep = False
                else:
                    recovery_started_at = None
                butt_wink = butt_wink_latched
                caution = caution_started_at is not None and now - caution_started_at >= args.hold_seconds * 0.6

                # Standing tuck uses the calibrated upright relationship as its
                # reference. Torso motion is gated so ordinary bending is not
                # mislabeled as an isolated pelvic tuck.
                standing_posterior_change = args.standing_posterior_sign * relative_angle
                standing_posterior_pelvis = (
                    args.standing_imu_sign
                    * args.pelvis_sign
                    * pelvis_delta
                )
                standing_warning_candidate = (
                    has_orientation and has_pose and not rep_active and standing
                    and abs(torso_delta) <= args.standing_max_torso
                    and standing_posterior_change >= args.standing_warning
                    and standing_posterior_pelvis >= args.standing_imu_warning
                )
                standing_caution_candidate = (
                    has_orientation and has_pose and not rep_active and standing
                    and abs(torso_delta) <= args.standing_max_torso
                    and standing_posterior_change >= args.standing_caution
                    and standing_posterior_pelvis >= args.standing_imu_warning * 0.6
                )
                standing_warning_started_at = (
                    now
                    if standing_warning_candidate and standing_warning_started_at is None
                    else standing_warning_started_at
                )
                standing_caution_started_at = (
                    now
                    if standing_caution_candidate and standing_caution_started_at is None
                    else standing_caution_started_at
                )
                if not standing_warning_candidate:
                    standing_warning_started_at = None
                if not standing_caution_candidate:
                    standing_caution_started_at = None
                standing_tuck_candidate = (
                    standing_warning_started_at is not None
                    and now - standing_warning_started_at >= args.hold_seconds
                )
                standing_caution = (
                    standing_caution_started_at is not None
                    and now - standing_caution_started_at >= args.hold_seconds * 0.6
                )
                if standing_tuck_candidate:
                    standing_tuck_latched = True
                    last_standing_tuck_at = now
                    standing_recovery_started_at = None

                standing_neutral_candidate = (
                    has_orientation
                    and has_pose
                    and not rep_active
                    and standing
                    and abs(torso_delta) <= args.standing_max_torso
                    and not standing_warning_candidate
                    and (
                        standing_posterior_change < args.standing_caution
                        or standing_posterior_pelvis <= args.recovery_angle
                    )
                )
                if standing_neutral_candidate:
                    if standing_recovery_started_at is None:
                        standing_recovery_started_at = now
                    elif now - standing_recovery_started_at >= args.recovery_seconds:
                        standing_tuck_latched = False
                        last_standing_tuck_at = None
                        standing_recovery_started_at = None
                else:
                    standing_recovery_started_at = None
                squat_active = rep_active
                standing_tuck = (
                    not squat_active and standing and standing_tuck_latched
                )

                if not has_orientation:
                    status = "RESULT: CANNOT JUDGE - NO IMU ORIENTATION"
                    color = (0, 0, 255)
                elif not has_pose:
                    status = "RESULT: CANNOT JUDGE - NO POSE"
                    color = (0, 220, 255)
                else:
                    # User-facing state machine: squat mode has absolute
                    # priority over all standing tuck state.
                    if squat_active:
                        if butt_wink:
                            status = "RESULT: BUTT WINK DETECTED"
                            color = (0, 0, 255)
                        else:
                            status = "RESULT: NORMAL"
                            color = (0, 255, 0)
                    else:
                        if standing_tuck:
                            status = "RESULT: STANDING PELVIC TUCK"
                            color = (0, 0, 255)
                        else:
                            status = "RESULT: STANDING NORMAL"
                            color = (0, 255, 0)

                lines = [
                    status,
                    verdict_reason(
                        has_orientation,
                        has_pose,
                        squat_active,
                        standing_tuck,
                        butt_wink,
                    ),
                    f"CALIBRATION DONE | Neutral raw pitch {neutral_pelvis:+.1f} deg | Neutral torso {neutral_torso:+.1f} deg",
                    f"Detection: quaternion PITCH {args.pelvis_sign:+d} | torso {args.torso_sign:+d} | squat posterior {args.posterior_sign:+d} | standing relative {args.standing_posterior_sign:+d} | standing IMU {args.standing_imu_sign:+d}",
                    f"IMU live ({imu.latest.angle_source}): pitch {imu.latest.pitch:+.1f} | roll {imu.latest.roll:+.1f} | yaw {imu.latest.yaw:+.1f} deg",
                    f"ACC: {imu.latest.acc_x:+.2f}, {imu.latest.acc_y:+.2f}, {imu.latest.acc_z:+.2f}",
                    f"QUAT: {imu.latest.quat_w:+.2f}, {imu.latest.quat_x:+.2f}, {imu.latest.quat_y:+.2f}, {imu.latest.quat_z:+.2f}",
                    f"IMU UDP: packets {imu.packet_count} | last {imu_age_text} | relative pitch: {pelvis_delta:+.1f} deg",
                    f"IMU raw: {shorten(imu.last_line, 48)}",
                    f"IMU parse: {imu.last_parse_error or 'OK'}",
                    f"Torso MediaPipe: {filtered_torso if has_pose and filtered_torso is not None else 0:.1f} deg | delta: {torso_delta:+.1f} | conf: {metrics['confidence'] if metrics else 0:.2f} | side: {metrics['tracking_side'] if metrics else '-'}",
                    f"Squat Active: {squat_active} | Descending: {descending_phase}",
                    f"Butt Wink State: {'BUTT WINK DETECTED' if butt_wink else 'NORMAL'}",
                    f"Relative Pelvis Angle: {relative_angle:+.1f} deg | Squat tuck: {pelvic_tuck_angle:+.1f} deg",
                    f"IMU pelvis posterior change: {pelvis_posterior_change:+.1f}/{args.pelvis_warning:.1f} deg",
                    f"Neutral recovery signal: {squat_recovery_angle:+.1f}/{args.recovery_angle:.1f} deg | holding: {recovery_started_at is not None}",
                    f"Hysteresis: ON >= {args.butt_wink_on_angle:.1f} deg/{args.butt_wink_on_seconds:.2f}s | OFF <= {args.recovery_angle:.1f} deg/{args.recovery_seconds:.2f}s",
                    f"Internal pre-warning: bottom={caution} | standing={standing_caution}",
                    f"Dynamic relative: {relative_angle:+.1f} | anterior ref: {anterior_reference if anterior_reference is not None else 0:+.1f} | reversal: {posterior_change:+.1f} deg",
                    f"Posterior velocity: {filtered_posterior_velocity:+.1f} deg/s | rep peak: {max_posterior_velocity:+.1f}/{args.posterior_rate_warning:.1f}",
                    f"Bottom IMU confirmation: {pelvis_posterior_change:+.1f}/{args.bottom_imu_warning:.1f} deg",
                    f"Recovery: signal <= {args.recovery_angle:.1f} deg for {args.recovery_seconds:.2f}s",
                    f"Standing posterior: {standing_posterior_change:+.1f} | tuck latched: {standing_tuck_latched} | pre-tucked rep: {pre_tucked_rep}",
                    f"Standing IMU posterior: {standing_posterior_pelvis:+.1f}/{args.standing_imu_warning:.1f} deg",
                    f"Standing tuck carry: {recent_standing_tuck if rep_active else standing_tuck_latched} | grace {args.standing_tuck_grace_seconds:.1f}s",
                    f"Knee: {knee:.1f} | knee v: {knee_velocity:+.1f} | torso v: {torso_velocity:+.1f} deg/s | bottom phase: {bottom_phase}",
                    "Rule: gradual requires relative+IMU pelvis change; bottom detector retained",
                ]
                display_frame = make_dashboard(frame, lines, color)

            cv2.imshow("SafeLift Python Butt Wink Monitor", display_frame)
            if front_frame is not None:
                front_display = front_frame.copy()
                front_tracking = front_results is not None and front_results.pose_landmarks is not None
                front_label = "FRONT MEDIAPIPE: TRACKING" if front_tracking else "FRONT MEDIAPIPE: WAITING"
                front_color = (0, 255, 0) if front_tracking else (0, 220, 255)
                cv2.rectangle(front_display, (8, 8), (355, 46), (0, 0, 0), -1)
                cv2.putText(
                    front_display,
                    front_label,
                    (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    front_color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("SafeLift Front Camera", front_display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                neutral_pelvis = None
                neutral_quaternion = None
                neutral_torso = None
                quaternion_samples = []
                pelvis_sum = 0.0
                pelvis_count = 0
                torso_sum = 0.0
                torso_count = 0
                calibration_start = time.time()
                calibration_collecting = False
                filtered_pelvis = None
                filtered_torso = None
                filtered_knee = None
                previous_relative = None
                previous_knee = None
                previous_torso = None
                previous_motion_time = None
                filtered_posterior_velocity = 0.0
                filtered_torso_velocity = 0.0
                rep_active = False
                anterior_reference = None
                descending_phase = False
                descent_reference = None
                pelvis_anterior_reference = None
                pelvic_tuck_angle = 0.0
                pelvis_posterior_change = 0.0
                gradual_butt_wink_started_at = None
                max_posterior_velocity = 0.0
                butt_wink_latched = False
                recovery_started_at = None
                warning_started_at = None
                caution_started_at = None
                standing_warning_started_at = None
                standing_caution_started_at = None
                standing_recovery_started_at = None
                standing_tuck_latched = False
                last_standing_tuck_at = None
                pre_tucked_rep = False
                print("Recalibrating...")
    finally:
        imu.stop()
        side_cap.release()
        if front_cap is not None:
            front_cap.release()
        pose.close()
        front_pose.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
