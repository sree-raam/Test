from ultralytics import YOLO
import cv2
from threading import Thread


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

    # Exit application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ---------------------------
# Release Resources
# ---------------------------
stream.stop()
cv2.destroyAllWindows()
