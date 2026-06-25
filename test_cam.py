import cv2


# ==========================================================
# RTSP Camera Preview Utility
#
# Description:
# This script connects to an RTSP camera stream,
# captures frames in real time, resizes them for display,
# and provides a simple live preview window.
#
# Purpose:
# - Verify camera connectivity
# - Test RTSP stream accessibility
# - Validate camera configuration
# - Debug industrial vision systems
#
# NOTE:
# Camera credentials and deployment-specific settings
# have been intentionally omitted from this public
# repository for security reasons.
#
# Replace the placeholder below with your own RTSP URL.
# ==========================================================


# ---------------------------
# Camera Configuration
# ---------------------------
camera1_url = "rtsp://USERNAME:PASSWORD@CAMERA_IP:554/stream1"


# ---------------------------
# Open Video Stream
# ---------------------------
cam1 = cv2.VideoCapture(camera1_url)


# ---------------------------
# Display Configuration
# ---------------------------
width = 640
height = 360


# ---------------------------
# Real-Time Camera Preview
# ---------------------------
while True:

    ret, frame = cam1.read()

    if not ret:
        print("Unable to read video stream.")
        break


    frame = cv2.resize(frame, (width, height))


    cv2.imshow(
        "RTSP Camera Preview",
        frame
    )


    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ---------------------------
# Cleanup
# ---------------------------
cam1.release()

cv2.destroyAllWindows()
