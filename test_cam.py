import cv2




# ------------------- CAMERA SETUP -------------------
camera1_url = "rtsp://IVY_SA:IVYSHAHALAM1234!@192.168.0.246:554/stream1"
# camera2_url = "rtsp://IVY_SA:IVYSHAHALAM1234!@192.168.0.156:554/stream1"

cam1 = cv2.VideoCapture(camera1_url)
# cam2 = cv2.VideoCapture(camera2_url)

width, height = 640, 360


while True:
    ret1, frame1 = cam1.read()
    # ret2, frame2 = cam2.read()

    # if not ret1 or not ret2:
    #     print("Error: Could not read from camera streams.")
    #     break

    frame1 = cv2.resize(frame1, (width, height))
    # frame2 = cv2.resize(frame2, (width, height))



    # combined = cv2.hconcat([frame1, frame2])
    # cv2.imshow("Bottle Tracking (Cam1 | Cam2)", combined)
    cv2.imshow("Camera iugjgjg", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam1.release()
# cam2.release()
cv2.destroyAllWindows()
