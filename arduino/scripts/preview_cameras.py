import argparse
import platform
import time

import cv2
import numpy as np


def backends():
    if platform.system().lower() == "windows":
        return [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)]
    return [("ANY", cv2.CAP_ANY)]


def open_camera(index: int, width: int, height: int):
    for backend_name, backend in backends():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        for _ in range(8):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap, backend_name
            time.sleep(0.03)

        cap.release()

    return None, ""


def label_frame(frame, label: str, width: int, height: int):
    if frame is None:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    cv2.rectangle(frame, (0, 0), (width, 38), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview OpenCV camera indexes in a tiled window.")
    parser.add_argument("--max-index", type=int, default=9)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args()

    cameras = []
    for index in range(args.max_index + 1):
        cap, backend_name = open_camera(index, args.width, args.height)
        if cap is not None:
            print(f"[{index}] opened backend={backend_name}")
            cameras.append((index, backend_name, cap))
        else:
            print(f"[{index}] unavailable")

    if not cameras:
        print("No cameras found.")
        return 1

    print("Press q or Esc to quit. Use the visible index number for --front_camera / --side_camera.")
    while True:
        tiles = []
        for index, backend_name, cap in cameras:
            ok, frame = cap.read()
            if not ok:
                frame = None
            tiles.append(label_frame(frame, f"CAM {index}  {backend_name}", args.width, args.height))

        while len(tiles) % 2 != 0:
            tiles.append(label_frame(None, "empty", args.width, args.height))

        rows = []
        for i in range(0, len(tiles), 2):
            rows.append(cv2.hconcat([tiles[i], tiles[i + 1]]))
        mosaic = cv2.vconcat(rows)
        cv2.imshow("Camera index preview", mosaic)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    for _, _, cap in cameras:
        cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
