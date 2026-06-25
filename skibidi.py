import cv2
import threading
import numpy as np
import time
import torch
from ultralytics import YOLO


# ==========================================================
# Multi-Camera YOLOv8 Detection with Threaded RTSP Streams
#
# Description:
# This script performs real-time YOLOv8 object detection
# on two independent camera streams using threaded video
# capture for smoother multi-camera monitoring.
#
# Features:
# • Multi-camera RTSP stream processing
# • Threaded frame capture
# • YOLOv8 object detection
# • GPU acceleration support
# • Split-screen visualization
# • Frame skipping for improved performance
#
# Applications:
# • Industrial monitoring
# • Inventory observation
# • Conveyor inspection
# • Warehouse vision systems
# • Real-time AI camera testing
#
# NOTE:
# Camera usernames, passwords, IP addresses, and deployment-
# specific details have been intentionally omitted from this
# public repository for security reasons.
#
# Replace the placeholders below with your own RTSP URLs.
# ==========================================================


# ---------------------------
# Device Configuration
# ---------------------------
if torch.cuda.is_available():
    device = 0
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("Running on CPU")


# ---------------------------
# Load YOLOv8 Model
# ---------------------------
model = YOLO("yolov8n.pt")

# Optional:
# model = YOLO("yolov8s.pt")  # Better accuracy with slightly lower FPS


# ---------------------------
# RTSP Camera Configuration
# ---------------------------
rtsp_urls = {
    "Camera 1": "rtsp://USERNAME:PASSWORD@CAMERA_IP_1:554/stream1",
    "Camera 2": "rtsp://USERNAME:PASSWORD@CAMERA_IP_2:554/stream1"
}


# ---------------------------
# Shared Frame Buffers
# ---------------------------
frames = {
    "Camera 1": None,
    "Camera 2": None
}

frame_locks = {
    "Camera 1": threading.Lock(),
    "Camera 2": threading.Lock()
}

stop_threads = False


# ---------------------------
# Capture and Detection Worker
# ---------------------------
def capture_and_detect(name, url, frame_skip=5):
    global frames, stop_threads

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"Cannot connect to {name}")
        return

    print(f"Connected to {name}")

    count = 0

    while not stop_threads:
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame from {name}")
            break

        frame = cv2.resize(frame, (640, 360))
        count += 1

        # Run detection on every Nth frame to reduce computation load
        if count % frame_skip == 0:
            results = model(
                frame,
                device=device,
                verbose=False
            )
            frame = results[0].plot()

        # Thread-safe frame update
        with frame_locks[name]:
            frames[name] = frame

    cap.release()


# ---------------------------
# Start Camera Threads
# ---------------------------
threads = []

for name, url in rtsp_urls.items():
    thread = threading.Thread(
        target=capture_and_detect,
        args=(name, url),
        daemon=True
    )
    thread.start()
    threads.append(thread)


# ---------------------------
# Display Loop
# ---------------------------
while True:
    frame_list = []

    for name in ["Camera 1", "Camera 2"]:
        with frame_locks[name]:
            if frames[name] is not None:
                frame_list.append(frames[name])

    if len(frame_list) == 2:
        combined_frame = np.hstack(frame_list)

        cv2.imshow(
            "Multi-Camera YOLOv8 Detection",
            combined_frame
        )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        stop_threads = True
        break

    # Prevent unnecessary CPU overload
    time.sleep(0.01)


# ---------------------------
# Cleanup
# ---------------------------
for thread in threads:
    thread.join()

cv2.destroyAllWindows()
