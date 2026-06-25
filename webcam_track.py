from ultralytics import YOLO
import cv2
from threading import Thread


# ==========================================================
# Real-Time YOLOv8 Object Tracking with ByteTrack
#
# Description:
# This script performs real-time object detection and
# multi-object tracking using YOLOv8 and ByteTrack.
#
# Features:
# • Threaded webcam video capture
# • YOLOv8 object detection
# • ByteTrack multi-object tracking
# • Persistent object identities
# • Annotated visualization
# • Low-latency video processing
#
# Applications:
# • Object tracking research
# • Smart surveillance systems
# • Warehouse monitoring
# • Industrial inspection
# • Computer vision prototyping
#
# Hardware:
# • Standard USB webcam
# • Laptop integrated camera
#
# NOTE:
# This implementation is intended for experimentation,
# benchmarking, and educational purposes.
#
# Press 'q' to terminate the application.
# ==========================================================


# ---------------------------
# Threaded Video Stream Class
# ---------------------------
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.running = True
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()


# ---------------------------
# Load YOLOv8 Detection Model
# ---------------------------
model = YOLO("yolov8n.pt")

# Optional:
# model = YOLO("yolov8s.pt")  # Improved accuracy with slightly lower FPS


# ---------------------------
# Initialize Video Stream
# ---------------------------
stream = VideoStream(0)


# ---------------------------
# Real-Time Detection Loop
# ---------------------------
while True:

    ret, frame = stream.read()

    if not ret:
        break

    # Perform object tracking using ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml"
    )

    # Generate annotated output frame
    annotated_frame = results[0].plot()

    cv2.imshow(
        "YOLOv8 Real-Time Object Tracking",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ---------------------------
# Release Resources
# ---------------------------
stream.stop()
cv2.destroyAllWindows()
