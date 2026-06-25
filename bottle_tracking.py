import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime


# ==========================================================
# Dual-Camera Bottle Inventory Counter using YOLOv8
#
# Description:
# This script performs bottle detection on two camera streams
# using YOLOv8 and estimates inventory changes based on
# bottle count differences between Camera 1 and Camera 2.
#
# Features:
# • Dual-camera RTSP stream processing
# • YOLOv8 object detection
# • Bottle-only counting
# • Inventory increase/decrease estimation
# • Split-screen visualization
# • Excel event logging
#
# Applications:
# • Conveyor inventory monitoring
# • Industrial process observation
# • Bottle flow analysis
# • Entry and exit counting
# • Computer vision prototyping
#
# NOTE:
# Camera usernames, passwords, IP addresses, company names,
# and deployment-specific details have been intentionally
# omitted from this public repository for security reasons.
#
# Replace the placeholders below with your own RTSP URLs.
# ==========================================================


# ---------------------------
# Load YOLOv8 Model
# ---------------------------
model = YOLO("yolov8x.pt")


# ---------------------------
# RTSP Camera Configuration
# ---------------------------
camera1_url = "rtsp://USERNAME:PASSWORD@CAMERA_IP_1:554/stream1"
camera2_url = "rtsp://USERNAME:PASSWORD@CAMERA_IP_2:554/stream1"


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
# Inventory Variables
# ---------------------------
inventory_count = 0
prev_cam1_count = 0
prev_cam2_count = 0
log_data = []


# ---------------------------
# Event Logger
# ---------------------------
def log_event(event, camera, remaining):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data.append(
        [
            timestamp,
            event,
            camera,
            "bottle",
            remaining
        ]
    )


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


    # Run YOLOv8 detection
    results1 = model(frame1)
    results2 = model(frame2)


    # Count bottle detections in both cameras
    cam1_bottle_count = sum(
        1
        for result in results1
        for box in result.boxes
        if model.names[int(box.cls[0])] == "bottle"
    )

    cam2_bottle_count = sum(
        1
        for result in results2
        for box in result.boxes
        if model.names[int(box.cls[0])] == "bottle"
    )


    # Estimate inventory increase from Camera 1
    if cam1_bottle_count > prev_cam1_count:
        added = cam1_bottle_count - prev_cam1_count
        inventory_count += added
        log_event("Added", "Camera 1", inventory_count)


    # Estimate inventory decrease from Camera 2
    if cam2_bottle_count > prev_cam2_count:
        removed = cam2_bottle_count - prev_cam2_count

        for _ in range(removed):
            if inventory_count > 0:
                inventory_count -= 1
                log_event("Removed", "Camera 2", inventory_count)


    prev_cam1_count = cam1_bottle_count
    prev_cam2_count = cam2_bottle_count


    # Split-screen display
    combined_frame = cv2.hconcat([frame1, frame2])

    cv2.putText(
        combined_frame,
        f"Estimated Inventory: {inventory_count}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow(
        "Dual-Camera Bottle Inventory Counter",
        combined_frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------
# Save Event Log
# ---------------------------
df = pd.DataFrame(
    log_data,
    columns=[
        "Time",
        "Event",
        "Camera",
        "Object",
        "Remaining"
    ]
)

df.to_excel(
    "bottle_tracking_log.xlsx",
    index=False
)


# ---------------------------
# Cleanup
# ---------------------------
vid1.release()
vid2.release()
cv2.destroyAllWindows()
