import cv2
from ultralytics import YOLO


# ---------------------------
# Load YOLOv8 Model
# ---------------------------
model = YOLO("yolov8x.pt")


# ---------------------------
# RTSP Camera Streams
# Replace with your own camera credentials
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


    # Object Detection
    results1 = model(frame1, stream=True)
    results2 = model(frame2, stream=True)


    # Camera 1 Detections
    for r in results1:
        for bbox in r.boxes:

            x1, y1, x2, y2 = map(int, bbox.xyxy[0])

            cls_idx = int(bbox.cls[0])
            cls_name = model.names[cls_idx]

            conf = round(float(bbox.conf[0]), 2)


            cv2.rectangle(
                frame1,
                (x1, y1),
                (x2, y2),
                (0,0,200),
                2
            )


            cv2.putText(
                frame1,
                f"{cls_name} {conf}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0,255,0),
                2
            )


    # Camera 2 Detections
    for r in results2:
        for bbox in r.boxes:

            x1,y1,x2,y2 = map(int,bbox.xyxy[0])

            cls_idx = int(bbox.cls[0])
            cls_name = model.names[cls_idx]

            conf = round(float(bbox.conf[0]),2)


            cv2.rectangle(
                frame2,
                (x1,y1),
                (x2,y2),
                (0,0,200),
                2
            )


            cv2.putText(
                frame2,
                f"{cls_name} {conf}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0,255,0),
                2
            )


    # Split-Screen View
    combined_frame = cv2.hconcat([frame1, frame2])



    cv2.imshow(
        "Dual Camera Object Detection",
        combined_frame
    )


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# ---------------------------
# Cleanup
# ---------------------------
vid1.release()
vid2.release()

cv2.destroyAllWindows()
