from picamera2 import Picamera2
import cv2
import numpy as np

# Define color boundaries for Rubik's cube colors (in HSV format)
# Adjust these based on your lighting conditions
COLOR_BOUNDS = {
    "red": [(0, 100, 100), (10, 255, 255)],
    "orange": [(10, 100, 100), (25, 255, 255)],
    "yellow": [(20, 100, 100), (40, 255, 255)],  # Adjusted for yellow
    "green": [(35, 100, 100), (85, 255, 255)],
    "blue": [(90, 100, 100), (130, 255, 255)],  # Adjusted for blue
    "white": [(0, 0, 200), (180, 50, 255)],  # Approximate for white
}


def detect_color(hsv, color_name, lower, upper):
    """Detect a specific color and return a mask."""
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    return mask


def draw_detected_colors(frame, hsv):
    """Draw rectangles around detected colors."""
    output = frame.copy()
    for color_name, (lower, upper) in COLOR_BOUNDS.items():
        mask = detect_color(hsv, color_name, lower, upper)

        # Find contours for the color
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return output


def white_balance(image):
    """Apply white balance correction using the gray-world algorithm."""
    img_float = np.float32(image)

    # Calculate the mean of each channel
    avg_b = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_r = np.mean(img_float[:, :, 2])

    # Calculate the average of all channels
    avg = (avg_b + avg_g + avg_r) / 3

    # Scale each channel based on the mean
    img_float[:, :, 0] = img_float[:, :, 0] * (avg / avg_b)  # Blue channel
    img_float[:, :, 1] = img_float[:, :, 1] * (avg / avg_g)  # Green channel
    img_float[:, :, 2] = img_float[:, :, 2] * (avg / avg_r)  # Red channel

    img_corrected = np.clip(img_float, 0, 255).astype(np.uint8)

    return img_corrected


# Initialize Picamera2
picam2 = Picamera2()

# Configure the camera to capture video (640x480 resolution)
video_config = picam2.create_video_configuration(main={"size": (640, 480)})
picam2.configure(video_config)

# Start the camera
picam2.start()

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Apply white balance correction
    balanced_frame = white_balance(frame)

    # Convert the balanced frame to HSV
    hsv = cv2.cvtColor(balanced_frame, cv2.COLOR_BGR2HSV)

    # Detect colors and draw on the frame
    output_frame = draw_detected_colors(balanced_frame, hsv)

    # Show the live video with color detection
    cv2.imshow("Rubik's Cube Color Detection", output_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the camera and close OpenCV windows
picam2.stop()
cv2.destroyAllWindows()
