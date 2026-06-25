import cv2
from ultralytics import YOLO


# ==========================================================
# Dual-Camera YOLOv8 Object Detection
#
# Description:
# This script performs real-time object detection using
# YOLOv8 on two independent RTSP camera streams and
# displays the annotated outputs in a split-screen view.
#
# Features:
# • Dual-camera monitoring
# • YOLOv8 object detection
# • Bounding box visualization
# • Confidence score display
# • Split-screen visualization
# • Real-time inference
#
# Applications:
# • Industrial monitoring
# • Inventory observation
# • Conveyor belt inspection
# • Warehouse surveillance
# • Multi-camera computer vision systems
#
# NOTE:
# Camera credentials, IP addresses, and deployment-
# specific settings have been intentionally omitted
# from this public repository for security reasons.
#
# Replace the placeholders below with your own
# RTSP camera configuration.
# ==========================================================


# ---------------------------
# Load YOLOv8 Model
# ---------------------------
model = YOLO("yolov8x.pt")


# ---------------------------
# RTSP Camera Configuration
# ---------------------------
camera1_url = "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1"
camera2_url = "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1"


# ---------------------------
# Open Video Streams
# ---------------------------
vid1 = cv2.VideoCapture(camera1_url)
vid2 = cv2.VideoCapture(camera2_url)


# ---------------------------
# Display Configuration
# ---------------------------
width, height = 640, 360


# ---------------------------
# Real-Time Detection Loop
# ---------------------------
while True:

    ret1, frame1 = vid1.read()
    ret2, frame2 = vid2.read()

    if not ret1 or not ret2:
        print("Error: Unable to read video streams.")
        break


    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))

    results1 = model(frame1, stream=True)
    results2 = model(frame2, stream=True)

    ...
