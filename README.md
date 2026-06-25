# AI Vision Detection Experiments

Computer vision experiments for real-time object detection, tracking, inventory monitoring, and bottle flow analysis using YOLOv8, OpenCV, RTSP camera streams, and GPU acceleration.

This repository serves as an experimental workspace for evaluating multiple approaches to bottle detection, object tracking, inventory counting, and industrial monitoring applications.

---

## Key Features

### Object Detection

- YOLOv8-based real-time object detection
- Bottle-only detection using COCO class labels
- GPU accelerated inference with CUDA support

### Object Tracking

- ByteTrack multi-object tracking
- Persistent object identity assignment
- Webcam and RTSP stream tracking

### Inventory Monitoring

- Dual-camera bottle counting
- Line-crossing event detection
- Remaining inventory estimation
- Entry and exit monitoring

### Industrial Monitoring

- Split-screen visualization
- RTSP camera integration
- Event logging to CSV and Excel
- Performance monitoring and FPS measurement

---

## Repository Structure

```text
AI-Vision-Detection/

├── AI_detection.py
├── bottle_tracking.py
├── bottle_tracking_gpu.py
├── webcam.py
├── webcam_track.py
├── test_cam.py
├── skibidi.py
└── README.md
```

---

## Technologies Used

- Python
- OpenCV
- YOLOv8
- Ultralytics
- PyTorch
- Pandas
- NumPy
- ByteTrack
- RTSP Streaming
- CUDA

---

## Main Experiments

### Dual-Camera Bottle Detection

Detect bottles from multiple RTSP cameras and display synchronized views.

### Inventory Tracking

Estimate bottle inventory based on object counts and crossing events.

### YOLOv8 Tracking

Evaluate multi-object tracking performance using ByteTrack.

### GPU Deployment

Assess real-time inference performance with CUDA acceleration.

### Data Logging

Store inventory events and monitoring results for reporting and analysis.

---

## Installation

Install required packages:

```bash
pip install ultralytics opencv-python torch pandas numpy
```

---

## Running Experiments

Bottle Tracking

```bash
python bottle_tracking.py
```

GPU Bottle Tracking

```bash
python bottle_tracking_gpu.py
```

Webcam Tracking

```bash
python webcam_track.py
```

Camera Testing

```bash
python test_cam.py
```

Press **q** to terminate the application.

---

## Notes

This repository contains experimental implementations developed for evaluating various computer vision techniques and industrial monitoring concepts. Some scripts may represent intermediate prototypes, testing utilities, or proof-of-concept implementations.

Future updates may consolidate the experiments into a unified vision-based intelligent monitoring framework.

---

## Author

**Sree Raam**

Bachelor of Electronic Systems Engineering (Hons.)

Malaysia-Japan International Institute of Technology (MJIIT)

Universiti Teknologi Malaysia
