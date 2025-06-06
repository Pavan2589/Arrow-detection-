# Arrow Detection and Distance Estimation

## Overview

This project implements an arrow detection system using a webcam feed. It detects arrow shapes within the camera frame, estimates their distance from the camera, and infers the arrow’s pointing direction (Left or Right). The system processes only a central Area of Interest (AOI) to improve accuracy and speed.

## Features

* Real-time capture from a specified webcam device (`/dev/videoX`)
* Camera calibration using a known diagonal Field of View (FoV) to calculate focal length
* Distance estimation to the arrow using the pinhole camera model
* Arrow shape detection via contour analysis and polygon approximation
* Arrow direction inference based on contour convexity and centroid analysis
* Visual feedback with contour outlines, bounding boxes, direction, and distance overlays

## How It Works

### 1. Camera Initialization

* Captures video frames at 1280x720 resolution using OpenCV’s `VideoCapture`.

### 2. Camera Calibration

* Calculates horizontal FoV and focal length from a known diagonal FoV (55°) and camera resolution.
* These parameters enable accurate distance estimation.

### 3. Distance Estimation

* Applies the pinhole camera model formula:

  ```
  distance = (real_arrow_width * focal_length) / pixel_width
  ```
* Includes a scaling factor to correct measurement inaccuracies.

### 4. Area of Interest (AOI)

* Focuses processing on the central 40% of each frame.
* Reduces false positives and speeds up detection.

### 5. Preprocessing Pipeline

* Converts the AOI frame to grayscale.
* Applies Gaussian blur to reduce noise.
* Detects edges using the Canny method.
* Enhances edges with dilation.

### 6. Arrow Shape Detection

* Finds contours and approximates them to polygons.
* Filters polygons by vertex count and aspect ratio to identify arrow shapes.
* Verifies arrow tip presence by analyzing distances from contour points to centroid.

### 7. Arrow Direction Inference

* Analyzes the convex hull of the contour.
* Compares centroid position with dense contour regions to classify direction as Left or Right.

### 8. Visualization and Output

* Draws contours and bounding boxes around detected arrows.
* Overlays direction and distance information as text on the frame.
* Displays a message if no arrow is detected.

## Requirements

* Python 3.x
* OpenCV (`opencv-python`)
* NumPy

## Usage

1. Connect your webcam and identify the device number (e.g., `/dev/video0`).
2. Run the script, specifying the webcam device if required.
3. The output window will show real-time arrow detection with distance and direction.
