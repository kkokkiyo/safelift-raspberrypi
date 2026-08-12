import argparse
import time

import cv2


def backends():
    return [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("ANY", cv2.CAP_ANY),
    ]


def try_camera(index: int, width: int, height: int):
    for backend_name, backend in backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        for _ in range(20):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap, backend_name, frame
            time.sleep(0.03)

        cap.release()

    return None, "", None


def main():
    parser = argparse.ArgumentParser(description="Preview OpenCV camera indices.")
    parser.add_argument("--max-index", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    print("Camera preview scan started.")
    print("A preview window will pop up for each working camera.")
    print("Press any key in the preview window to move to the next camera.")
    print()

    working = []
    for index in range(args.max_index + 1):
        print(f"Testing camera index {index}...")
        cap, backend_name, frame = try_camera(index, args.width, args.height)
        if cap is None:
            print(f"[{index}] unavailable")
            continue

        h, w = frame.shape[:2]
        print(f"[{index}] OK backend={backend_name} frame={w}x{h}")
        working.append((index, backend_name))

        start = time.time()
        while time.time() - start < args.seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            cv2.putText(
                frame,
                f"CAMERA INDEX {index} / {backend_name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Camera Preview Scanner", frame)
            if cv2.waitKey(1) & 0xFF != 255:
                break

        cap.release()

    cv2.destroyAllWindows()
    print()
    print("Working cameras:")
    for index, backend_name in working:
        print(f"  {index}: {backend_name}")

    if not working:
        print("  none")


if __name__ == "__main__":
    main()
