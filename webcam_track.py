from ultralytics import YOLO
import cv2
from threading import Thread

# Threaded Video Stream Class
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

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy (slightly slower)

# Start threaded video stream
stream = VideoStream(0)

while True:
    ret, frame = stream.read()
    if not ret:
        break

    # Run YOLO tracking
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")  # Built-in tracker

    # Annotate frame
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Tracking (Optimized)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()
