# Face Recognition Attendance System

This project is a web-based attendance system that uses face recognition to mark attendance automatically. Built with Python, Flask, and OpenCV, it detects faces via a webcam, identifies registered users using a K-Nearest Neighbors (KNN) classifier, and logs attendance in a CSV file. The system includes a web interface for viewing attendance records and adding new users.

## Features
- **Face Recognition**: Identifies registered users using a pre-trained KNN model and OpenCV's Haar Cascade classifier for face detection.
- **Attendance Tracking**: Logs attendance with name, roll number, and timestamp in a daily CSV file (`Attendance/Attendance-MM_DD_YY.csv`).
- **User Registration**: Allows adding new users by capturing face images via webcam and training the recognition model.
- **Web Interface**: Provides a Flask-based web interface to view attendance records and manage users.
- **Error Handling**: Includes robust error handling for webcam access, file operations, and face detection failures.
- **Object-Oriented Design**: Uses a modular `AttendanceSystem` class to encapsulate all functionality, improving maintainability.

## Prerequisites
To run this project, ensure you have the following:
- **Python 3.6+**
- **Webcam** (for face detection and user registration)
- **Dependencies**:
  - `opencv-python` (for face detection and image processing)
  - `numpy` (for array operations)
  - `scikit-learn` (for KNN classifier)
  - `pandas` (for CSV handling)
  - `joblib` (for model serialization)
  - `flask` (for the web interface)
- **Haar Cascade File**: `haarcascade_frontalface_default.xml` (included with OpenCV or downloadable from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades))
- **Optional**: A `background.png` image (1280x720 or larger) for overlaying webcam feed. If not provided, a black fallback image is used.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
