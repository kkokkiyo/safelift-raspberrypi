"""Show MediaPipe Pose landmarks on a DroidCam video feed.

Examples (run from the Backend directory):
    python scripts/droidcam_mediapipe.py
    python scripts/droidcam_mediapipe.py --camera 1
    python scripts/droidcam_mediapipe.py --ip 192.168.0.20
"""

import argparse
import platform
import time

import cv2
import mediapipe as mp


def open_camera(index: int, width: int, height: int):
    """Open a local camera, preferring DirectShow for DroidCam on Windows."""
    backends = [cv2.CAP_ANY]
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for backend in backends:
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened():
            capture.release()
            continue

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, 30)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # A virtual camera can take a moment before returning its first frame.
        deadline = time.time() + 3.0
        while time.time() < deadline:
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                return capture
            time.sleep(0.05)
        capture.release()

    raise RuntimeError(
        f"Camera index {index} could not be opened. "
        "Run scripts/preview_cameras.py and pass the DroidCam number with --camera."
    )


def open_ip_stream(ip: str, port: int):
    """Open DroidCam's Wi-Fi MJPEG stream."""
    urls = [
        f"http://{ip}:{port}/video",
        f"http://{ip}:{port}/mjpegfeed",
    ]
    for url in urls:
        capture = cv2.VideoCapture(url)
        if capture.isOpened():
            ok, frame = capture.read()
            if ok and frame is not None and frame.size:
                print(f"DroidCam stream: {url}")
                return capture
        capture.release()
    raise RuntimeError(
        f"DroidCam stream at {ip}:{port} could not be opened. "
        "Check the IP shown in the DroidCam phone app."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display MediaPipe Pose landmarks using a DroidCam feed."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=2,
        help="DroidCam virtual-camera index (default: 2).",
    )
    parser.add_argument(
        "--ip",
        default="",
        help="Phone IP shown in DroidCam; when set, the IP stream is used.",
    )
    parser.add_argument("--port", type=int, default=4747)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Mirror the image horizontally.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capture = (
        open_ip_stream(args.ip, args.port)
        if args.ip
        else open_camera(args.camera, args.width, args.height)
    )

    pose_module = mp.solutions.pose
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    pose = pose_module.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    print("DroidCam MediaPipe started. Press Q or Esc to quit.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("DroidCam frame could not be read.")
                break

            if args.flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)

            if result.pose_landmarks:
                drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    pose_module.POSE_CONNECTIONS,
                    landmark_drawing_spec=styles.get_default_pose_landmarks_style(),
                )
                status, color = "POSE DETECTED", (0, 255, 0)
            else:
                status, color = "NO POSE", (0, 0, 255)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"DroidCam | {status} | {smoothed_fps:.1f} FPS",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("DroidCam MediaPipe Pose", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        pose.close()
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
