import cv2
import numpy as np
import math

# Use the appropriate camera index, based on your system
camera_index = 4  # Corresponding to /dev/video5

# Initialize webcam
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Webcam at /dev/video{camera_index} not accessible.")
    exit()

# Webcam Specifications
RESOLUTION_WIDTH = 1280  # Horizontal resolution of 720p
RESOLUTION_HEIGHT = 720  # Vertical resolution of 720p
DIAGONAL_FOV = 55  # Diagonal Field of View in degrees
REAL_ARROW_WIDTH = 17.0  # Real-world arrow width in cm

# Distance scaling factor (calibrated to correct measurement errors)
SCALING_FACTOR = 1.67

# Compute horizontal FoV (hFoV) and focal length
aspect_ratio = 16 / 9
hFoV = 2 * math.degrees(
    math.atan((16 / math.sqrt(16**2 + 9**2)) * math.tan(math.radians(DIAGONAL_FOV / 2)))
)
focal_length = RESOLUTION_WIDTH / (2 * math.tan(math.radians(hFoV / 2)))

# Define Area of Interest (AOI)
AOI_X_START = 0.3  # Narrowed from 0.2 for a smaller AOI
AOI_X_END = 0.7    # Narrowed from 0.8
AOI_Y_START = 0.3  # Narrowed from 0.2
AOI_Y_END = 0.7    # Narrowed from 0.8

def is_arrow_shape(contour, approx):
    """
    Check if a contour has the shape of an arrow.
    """
    if len(approx) < 7 or len(approx) > 10:  # Adjust for tighter vertex count
        return False

    # Compute the bounding rectangle
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = w / float(h)

    if not (1.5 < aspect_ratio < 3.5):  # Arrows typically have this aspect ratio
        return False

    # Calculate centroid
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return False
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Verify the arrow tip (farthest point from the centroid)
    vertices = approx.reshape(-1, 2)
    tip = max(vertices, key=lambda point: np.linalg.norm(point - np.array([cx, cy])))

    # Check if the tip is significantly farther than other vertices
    tip_distance = np.linalg.norm(tip - np.array([cx, cy]))
    avg_distance = np.mean([np.linalg.norm(v - np.array([cx, cy])) for v in vertices])
    
    if tip_distance < 1.5 * avg_distance:  # The tip must be a prominent outlier
        return False

    return True

def detect_arrow_direction(contour,approx):
    """
    Determine the direction of the arrow based on the densest part of the contour.
    """
    # Calculate the convex hull of the contour
    hull = cv2.convexHull(contour)

    # Compute the centroid of the contour
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return "Unknown"
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Calculate distances of hull points from the centroid
    distances = [np.linalg.norm(np.array([cx, cy]) - point[0]) for point in hull]

    # Identify the densest part based on minimal distances
    dense_point_index = np.argmin(distances)
    dense_point = hull[dense_point_index][0]

    # Compare dense point's x-coordinate with centroid's x-coordinate to determine direction
    if dense_point[0] > cx:
        return "Left"
    else:
        return "Right"

def calculate_distance(pixel_width):
    """
    Calculate the corrected distance using the scaling factor.
    """
    if pixel_width == 0:
        return None  # Avoid division by zero
    measured_distance = (REAL_ARROW_WIDTH * focal_length) / pixel_width
    corrected_distance = measured_distance * SCALING_FACTOR
    return corrected_distance

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Frame capture from /dev/video{camera_index} failed.")
        break

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (RESOLUTION_WIDTH, RESOLUTION_HEIGHT))
    height, width = frame.shape[:2]

    # Define AOI coordinates
    x_start = int(AOI_X_START * width)
    x_end = int(AOI_X_END * width)
    y_start = int(AOI_Y_START * height)
    y_end = int(AOI_Y_END * height)

    # Draw AOI for visualization
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection with adjusted thresholds
    edges = cv2.Canny(blurred, 30, 200)

    # Perform dilation to connect fragmented edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours with hierarchical retrieval
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    arrow_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # Adjust contour area threshold
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        x, y, w, h = cv2.boundingRect(approx)
        if not (x_start <= x and x + w <= x_end and y_start <= y and y + h <= y_end):
            continue

        if is_arrow_shape(contour, approx):
            direction = detect_arrow_direction(contour, approx)

            # Calculate and scale distance
            distance = calculate_distance(w)
            if distance:
                distance_text = f"Distance: {distance:.2f} cm"
            else:
                distance_text = "Distance: Unknown"

            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"Arrow: {direction}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
            cv2.putText(frame, distance_text, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            arrow_detected = True
            break

    if not arrow_detected:
        cv2.putText(frame, "No Arrow Detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow("Webcam Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
