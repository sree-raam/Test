import cv2
import torch
from ultralytics import YOLO
import pandas as pd
import numpy as np
from datetime import datetime
import time
from threading import Thread


# ==========================================================
# Dual-Camera Bottle Segmentation and Line-Crossing Counter
#
# Description:
# This script performs real-time bottle segmentation using
# YOLOv8 segmentation on two camera streams. It estimates
# object movement across a virtual counting line and logs
# inventory count changes into a CSV file.
#
# Features:
# • Dual-camera RTSP stream processing
# • Threaded camera capture
# • YOLOv8 segmentation
# • Bottle-only detection using COCO class 39
# • Virtual line-crossing count logic
# • Real-time inventory count estimation
# • Contrast enhancement for one camera stream
# • FPS display
# • CSV logging
#
# Applications:
# • Conveyor monitoring
# • Inventory flow analysis
# • Industrial process observation
# • Object movement estimation
# • AI vision system prototyping
#
# NOTE:
# Camera usernames, passwords, IP addresses, company names,
# and deployment-specific details have been intentionally
# omitted from this public repository for security reasons.
#
# Replace the placeholders below with your own RTSP URLs.
# ==========================================================


# ---------------------------
# Threaded Camera Class
# ---------------------------
class CameraStream:
    def __init__(self, src, width=640, height=480):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.thread = Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            self.ret, self.frame = self.cap.read()

    def read(self):
        if self.ret and self.frame is not None:
            return cv2.resize(self.frame, (self.width, self.height))
        return None

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()


# ---------------------------
# Helper Functions
# ---------------------------
def draw_segmentation_masks(image, result, color=(0, 255, 0)):
    if result.masks is None:
        return image

    masks = result.masks.data.cpu().numpy()

    for mask in masks:
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            cv2.drawContours(
                image,
                [contour],
                -1,
                color,
                thickness=4
            )

    return image


def put_text_bottom_right(frame, text, y_offset, color, scale=0.9):
    text_size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        2
    )

    x = frame.shape[1] - text_size[0] - 10
    y = frame.shape[0] - y_offset

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness=3
    )


def enhance_contrast(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    l_channel = cv2.equalizeHist(l_channel)

    enhanced_lab = cv2.merge(
        (l_channel, a_channel, b_channel)
    )

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )


# ---------------------------
# GPU and Model Setup
# ---------------------------
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8s-seg.pt")


# ---------------------------
# RTSP Camera Configuration
# ---------------------------
camera1_url = "rtsp://USERNAME:PASSWORD@CAMERA_IP_1:554/stream2"
camera2_url = "rtsp://USERNAME:PASSWORD@CAMERA_IP_2:554/stream2"


cam1 = CameraStream(camera1_url)
cam2 = CameraStream(camera2_url)


# ---------------------------
# Runtime Variables
# ---------------------------
csv_file = "bottle_count_log.csv"

prev_time = time.time()
inventory_count = 0
last_logged_count = None

# Format:
# {
#   camera_id: {
#       object_id: last_x_position
#   }
# }
line_positions = {}


# Create CSV header
pd.DataFrame(
    columns=["Timestamp", "Inventory_Count"]
).to_csv(
    csv_file,
    index=False
)


try:
    while True:

        frame1 = cam1.read()
        frame2 = cam2.read()

        if frame1 is None or frame2 is None:
            continue


        # Optional contrast enhancement for second camera stream
        frame2 = enhance_contrast(frame2)


        # Detect bottle class only
        results = model.predict(
            [frame1, frame2],
            conf=0.1,
            iou=0.45,
            classes=[39],
            device=device,
            verbose=False
        )


        cam_results = [results[0], results[1]]

        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()

        annotated_frames = []


        for cam_index, (frame, result) in enumerate(
            zip([frame1, frame2], cam_results),
            start=1
        ):

            annotated = draw_segmentation_masks(
                frame.copy(),
                result
            )


            height, width = annotated.shape[:2]
            line_x = width // 2

            cv2.line(
                annotated,
                (line_x, 0),
                (line_x, height),
                (255, 0, 0),
                2
            )


            if cam_index not in line_positions:
                line_positions[cam_index] = {}


            if result.boxes is not None:
                for object_index, box in enumerate(
                    result.boxes.xyxy.cpu().numpy()
                ):

                    x1, y1, x2, y2 = box

                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Note:
                    # This is a simple frame-level object index.
                    # For robust tracking, use persistent tracking IDs.
                    object_id = object_index

                    last_x = line_positions[cam_index].get(object_id)


                    if last_x is not None:

                        # Camera 1 counts objects entering the monitored zone
                        if cam_index == 1 and last_x < line_x <= center_x:
                            inventory_count += 1

                        # Camera 2 counts objects leaving the monitored zone
                        elif cam_index == 2 and last_x > line_x >= center_x:
                            inventory_count = max(
                                0,
                                inventory_count - 1
                            )


                    line_positions[cam_index][object_id] = center_x

                    cv2.circle(
                        annotated,
                        (center_x, center_y),
                        5,
                        (0, 0, 255),
                        -1
                    )


            put_text_bottom_right(
                annotated,
                f"Camera {cam_index}",
                90,
                (0, 255, 0)
            )


            if cam_index == 1:
                put_text_bottom_right(
                    annotated,
                    f"Inventory Count: {inventory_count}",
                    60,
                    (0, 0, 255)
                )

                put_text_bottom_right(
                    annotated,
                    f"FPS: {fps:.2f}",
                    30,
                    (255, 255, 0)
                )


            annotated_frames.append(annotated)


        # Match frame sizes before horizontal concatenation
        if annotated_frames[0].shape != annotated_frames[1].shape:

            target_height = min(
                annotated_frames[0].shape[0],
                annotated_frames[1].shape[0]
            )

            annotated_frames[0] = cv2.resize(
                annotated_frames[0],
                (
                    int(
                        annotated_frames[0].shape[1]
                        * target_height
                        / annotated_frames[0].shape[0]
                    ),
                    target_height
                )
            )

            annotated_frames[1] = cv2.resize(
                annotated_frames[1],
                (
                    int(
                        annotated_frames[1].shape[1]
                        * target_height
                        / annotated_frames[1].shape[0]
                    ),
                    target_height
                )
            )


        combined = cv2.hconcat(annotated_frames)

        cv2.imshow(
            "Bottle Segmentation and Line-Crossing Counter",
            combined
        )


        # Log only when count changes
        if last_logged_count != inventory_count:

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            pd.DataFrame(
                [[timestamp, inventory_count]],
                columns=["Timestamp", "Inventory_Count"]
            ).to_csv(
                csv_file,
                mode="a",
                header=False,
                index=False
            )

            last_logged_count = inventory_count


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


finally:
    cam1.stop()
    cam2.stop()
    cv2.destroyAllWindows()

    print(
        f"Detection session ended. Data saved to {csv_file}"
    )
