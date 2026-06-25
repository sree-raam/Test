import cv2
import time
import torch
import pandas as pd
from datetime import datetime
from ultralytics import YOLO


# ==========================================================
# General AI Detection System using YOLOv8
#
# Description:
# This script performs real-time object detection using YOLOv8
# on a webcam, video file, or RTSP camera stream.
#
# Features:
# • Real-time object detection
# • Webcam / video / RTSP support
# • GPU acceleration if CUDA is available
# • FPS monitoring
# • Detection confidence display
# • CSV event logging
# • Safe public configuration with no credentials
#
# NOTE:
# Camera usernames, passwords, IP addresses, and deployment-
# specific details are intentionally omitted.
# ==========================================================


# ---------------------------
# Configuration
# ---------------------------
MODEL_PATH = "yolov8n.pt"

# Use 0 for webcam.
# Replace with your own video path or RTSP URL if needed.
SOURCE = 0
# SOURCE = "sample_video.mp4"
# SOURCE = "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1"

CONFIDENCE_THRESHOLD = 0.35
LOG_FILE = "ai_detection_log.csv"


# ---------------------------
# Device Setup
# ---------------------------
if torch.cuda.is_available():
    DEVICE = 0
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("Using CPU")


# ---------------------------
# Load YOLOv8 Model
# ---------------------------
model = YOLO(MODEL_PATH)


# ---------------------------
# Open Video Source
# ---------------------------
cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    raise RuntimeError("Unable to open video source.")


# ---------------------------
# Create Log File
# ---------------------------
pd.DataFrame(
    columns=["Timestamp", "Class", "Confidence", "X1", "Y1", "X2", "Y2"]
).to_csv(LOG_FILE, index=False)


prev_time = time.time()


# ---------------------------
# Real-Time Detection Loop
# ---------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("End of stream or failed to read frame.")
        break

    start_time = time.time()

    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        verbose=False
    )

    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            detections.append([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                class_name,
                round(confidence, 3),
                x1,
                y1,
                x2,
                y2
            ])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{class_name} {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    if detections:
        pd.DataFrame(
            detections,
            columns=["Timestamp", "Class", "Confidence", "X1", "Y1", "X2", "Y2"]
        ).to_csv(
            LOG_FILE,
            mode="a",
            header=False,
            index=False
        )

    fps = 1 / (time.time() - start_time)

    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.imshow("General AI Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------
# Cleanup
# ---------------------------
cap.release()
cv2.destroyAllWindows()

print(f"Detection completed. Log saved to {LOG_FILE}")
