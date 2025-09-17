import cv2
import threading
import numpy as np
import time
import torch
from ultralytics import YOLO

# ✅ Check if GPU is available
if torch.cuda.is_available():
    device = 0
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("⚠ Running on CPU (slower)")

# ✅ Load YOLOv8 model
model = YOLO('yolov8n.pt')  # use yolov8s.pt for better accuracy

# ✅ Camera RTSP URLs
rtsp_urls = {
    "Camera 1": "rtsp://IVY_SA:IVYSHAHALAM1234!@192.168.0.246:554/stream1",
    "Camera 2": "rtsp://IVY_SA:IVYSHAHALAM1234!@192.168.0.156:554/stream1"
}

frames = {"Camera 1": None, "Camera 2": None}
frame_locks = {"Camera 1": threading.Lock(), "Camera 2": threading.Lock()}
stop_threads = False

# ✅ Capture & Detect Function
def capture_and_detect(name, url, frame_skip=5):
    global frames, stop_threads
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"❌ Cannot connect {name}")
        return
    print(f"✅ Connected {name}")

    count = 0
    while not stop_threads:
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed frame {name}")
            break

        frame = cv2.resize(frame, (640, 360))
        count += 1

        # Process every Nth frame
        if count % frame_skip == 0:
            results = model(frame, device=device, verbose=False)
            frame = results[0].plot()

        # Thread-safe update
        with frame_locks[name]:
            frames[name] = frame

    cap.release()

# ✅ Start threads
threads = []
for name, url in rtsp_urls.items():
    t = threading.Thread(target=capture_and_detect, args=(name, url))
    t.start()
    threads.append(t)

# ✅ Display Loop
while True:
    frame_list = []
    for name in ["Camera 1", "Camera 2"]:
        with frame_locks[name]:
            if frames[name] is not None:
                frame_list.append(frames[name])

    if len(frame_list) == 2:
        combined_frame = np.hstack(frame_list)
        cv2.imshow("Tapo YOLOv8 AI Detection (GPU)", combined_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_threads = True
        break

    time.sleep(0.01)  # Prevent CPU overload

# ✅ Cleanup
for t in threads:
    t.join()
cv2.destroyAllWindows()
