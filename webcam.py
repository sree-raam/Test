import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8x.pt')

# Tapo Cameras RTSP URLs
camera1_url = "rtsp://IVY_SA:IVYSHAHALAM1234!@192.168.0.246:554/stream1"
camera2_url = "rtsp://IVY_SA:IVYSHAHALAM1234!@192.168.0.156:554/stream1"

# Open both streams
vid1 = cv2.VideoCapture(camera1_url)
vid2 = cv2.VideoCapture(camera2_url)

# Define display size for each camera
width, height = 640, 360  # Resize for split screen

while True:
    ret1, frame1 = vid1.read()
    ret2, frame2 = vid2.read()

    if not ret1 or not ret2:
        print("Error: Could not read one or both camera streams.")
        break

    # Resize frames for split screen
    frame1 = cv2.resize(frame1, (width, height))
    frame2 = cv2.resize(frame2, (width, height))

    # Run YOLO on both frames
    results1 = model(frame1, stream=True)
    results2 = model(frame2, stream=True)

    # Draw detections on frame1
    for r in results1:
        for bbox in r.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            cls_idx = int(bbox.cls[0])
            cls_name = model.names[cls_idx]
            conf = round(float(bbox.conf[0]), 2)
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 0, 200), 2)
            cv2.putText(frame1, f'{cls_name} {conf}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

    # Draw detections on frame2
    for r in results2:
        for bbox in r.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            cls_idx = int(bbox.cls[0])
            cls_name = model.names[cls_idx]
            conf = round(float(bbox.conf[0]), 2)
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 0, 200), 2)
            cv2.putText(frame2, f'{cls_name} {conf}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)

    # Combine frames horizontally (side by side)
    combined_frame = cv2.hconcat([frame1, frame2])

    # Show the combined frame
    cv2.imshow('Tapo Cameras Split View', combined_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid1.release()
vid2.release()
cv2.destroyAllWindows()
