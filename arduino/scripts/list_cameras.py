import cv2
import platform
import time


def backends():
    if platform.system().lower() == "windows":
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
    return [("ANY", cv2.CAP_ANY)]


def read_warm_frame(cap, attempts=8):
    frame = None
    for _ in range(attempts):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True, frame
        time.sleep(0.03)
    return False, frame


def main() -> int:
    print("Scanning OpenCV camera indexes 0-9 with multiple Windows backends...")
    for index in range(10):
        found = False
        for backend_name, backend in backends():
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, frame = read_warm_frame(cap)
            if ok and frame is not None and frame.size > 0:
                height, width = frame.shape[:2]
                real_backend = cap.getBackendName() if hasattr(cap, "getBackendName") else backend_name
                print(f"[{index}] OK {width}x{height} backend={real_backend}")
                found = True
                cap.release()
                break

            print(f"[{index}] {backend_name} opened but no frame")
            cap.release()
        if not found:
            print(f"[{index}] unavailable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
