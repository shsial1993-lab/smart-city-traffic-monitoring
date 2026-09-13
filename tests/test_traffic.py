import cv2
import numpy as np

from src.traffic import VehicleDetector


def test_detector_finds_two_blobs() -> None:
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.rectangle(frame, (5, 10), (35, 40), (255, 255, 255), -1)
    cv2.rectangle(frame, (85, 50), (125, 80), (255, 255, 255), -1)
    assert len(VehicleDetector(min_area=50).detect(frame)) == 2
