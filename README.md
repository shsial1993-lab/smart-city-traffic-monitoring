# Smart City Traffic Monitoring

A lightweight OpenCV baseline for vehicle-like object detection in fixed-camera
scenes. It uses thresholding, morphology, and contour filtering so the full demo runs
without downloading model weights. The detector can later be replaced with a YOLO or
ONNX Runtime backend while keeping the tracking interface stable.

## What it demonstrates

- Frame preprocessing and noise cleanup
- Bounding-box extraction from connected contours
- A small typed detection interface
- A deterministic synthetic-frame demo

## Run

~~~bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m src.demo
~~~

This is a baseline for experimentation. Real deployments need camera calibration,
privacy safeguards, tracking, weather/lighting tests, and a validated detector.

## Structure

- src/traffic.py — contour-based vehicle detector
- src/demo.py — synthetic traffic frame

## License

Apache-2.0
