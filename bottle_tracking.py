import cv2
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

# Load YOLO model
model = YOLO('yolov8x.pt')

# Camera RTSP URLs
camera1_url = "rtsp://IVY_SA:IVYSHAHALAM1234!@192.168.0.246:554/stream1"
camera2_url = "rtsp://IVY_SA:IVYSHAHALAM1234!@192.168.0.156:554/stream1"

# Open video streams
vid1 = cv2.VideoCapture(camera1_url)
vid2 = cv2.VideoCapture(camera2_url)

width, height = 640, 360

# Track bottles
inventory_count = 0
prev_cam1 = 0
prev_cam2 = 0
log_data = []

# Log function
def log_event(event, camera, remaining):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data.append([timestamp, event, camera, "bottle", remaining])

while True:
    ret1, frame1 = vid1.read()
    ret2, frame2 = vid2.read()

    if not ret1 or not ret2:
        print("Error: Could not read streams")
        break

    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))

    # Detect objects
    results1 = model(frame1)
    results2 = model(frame2)

    # Count bottles in both cameras
    cam1_bottle_count = sum(1 for r in results1 for b in r.boxes if model.names[int(b.cls[0])] == "bottle")
    cam2_bottle_count = sum(1 for r in results2 for b in r.boxes if model.names[int(b.cls[0])] == "bottle")

    # Add new bottles from Camera 1 (only when count increases)
    if cam1_bottle_count > prev_cam1:
        added = cam1_bottle_count - prev_cam1
        inventory_count += added
        log_event("Added", "Camera 1", inventory_count)

    # Remove bottles from Camera 2 (only when count increases)
    if cam2_bottle_count > prev_cam2:
        removed = cam2_bottle_count - prev_cam2
        for _ in range(removed):
            if inventory_count > 0:
                inventory_count -= 1
                log_event("Removed", "Camera 2", inventory_count)

    prev_cam1 = cam1_bottle_count
    prev_cam2 = cam2_bottle_count

    # Show combined frames
    combined_frame = cv2.hconcat([frame1, frame2])
    cv2.putText(combined_frame, f"Remaining bottles: {inventory_count}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Bottle Tracking', combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save logs to Excel
df = pd.DataFrame(log_data, columns=["Time", "Event", "Camera", "Object", "Remaining"])
df.to_excel("bottle_tracking_log.xlsx", index=False)

vid1.release()
vid2.release()
cv2.destroyAllWindows()
