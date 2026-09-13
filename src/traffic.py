from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    width: int
    height: int
    area: float

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


class VehicleDetector:
    """Contour detector for fixed-camera prototyping."""

    def __init__(self, threshold: int = 180, min_area: int = 150) -> None:
        self.threshold = threshold
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim not in (2, 3):
            raise ValueError('frame must be grayscale or color')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            detections.append(Detection(x, y, width, height, area))
        return sorted(detections, key=lambda item: (item.y, item.x))
