from __future__ import annotations

import cv2
import numpy as np

from .traffic import VehicleDetector


def main() -> None:
    frame = np.zeros((240, 420, 3), dtype=np.uint8)
    cv2.rectangle(frame, (35, 80), (145, 150), (255, 255, 255), -1)
    cv2.rectangle(frame, (230, 55), (370, 135), (255, 255, 255), -1)
    detections = VehicleDetector(min_area=200).detect(frame)
    print('detected objects:', len(detections))
    for detection in detections:
        print('bbox:', (detection.x, detection.y, detection.width, detection.height))


if __name__ == '__main__':
    main()
