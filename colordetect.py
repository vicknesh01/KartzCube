# This module is used to detect the color and save the values
# Created by Vickneshwaran Elangkeeran



from picamera2 import Picamera2
import cv2
import imutils
import numpy as np

print("KartzCube Color Detector. Powered by OpenCV " + cv2.__version__)







# Initialize the camera
picam2 = Picamera2()

# Configure the camera
camera_config = picam2.create_still_configuration()
picam2.configure(camera_config)

try:
    # Start the camera
    picam2.start()

    # Capture the picture
    picam2.capture_file('test4.jpg')

    print("Image taken using Camera")

finally:
    # Stop the camera
    picam2.stop()






def is_approx_3x3_layout(squares, tolerance=50):
    """
    Checks if the detected squares approximately form a 3x3 grid layout.
    Args:
        squares: List of square contours.
        tolerance: Pixel tolerance for alignment.
    Returns:
        Tuple (grid_centers, success) where:
        - grid_centers: List of interpolated grid centers.
        - success: True if the structure is a valid 3x3 grid (even partially).
    """
    if len(squares) < 4:
        # Too few squares to form a grid
        return [], False

    # Extract bounding box centers of each square
    centers = []
    for square in squares:
        x, y, w, h = cv2.boundingRect(square)
        centers.append((x + w // 2, y + h // 2))  # Center of the square

    print("Centers detected:", centers)  # Debug: print detected centers

    # Sort centers by y-coordinate (row-wise alignment)
    centers = sorted(centers, key=lambda c: c[1])

    # Group into rows based on proximity of y-coordinates
    rows = []
    current_row = [centers[0]]
    for i in range(1, len(centers)):
        if abs(centers[i][1] - current_row[0][1]) <= tolerance:
            current_row.append(centers[i])
        else:
            rows.append(sorted(current_row, key=lambda c: c[0]))  # Sort row by x-coordinate
            current_row = [centers[i]]
    rows.append(sorted(current_row, key=lambda c: c[0]))

    print("Rows detected:", rows)  # Debug: print detected rows

    # Ensure at most 3 rows
    if len(rows) > 3:
        return [], False

    # Ensure rows have at most 3 squares, pad with None if necessary
    grid_centers = []
    for row in rows:
        while len(row) < 3:
            row.append(None)  # Placeholder for missing squares
        grid_centers.append(row[:3])  # Trim to exactly 3 elements

    # Interpolate missing rows if less than 3 rows are detected
    while len(grid_centers) < 3:
        grid_centers.append([None, None, None])  # Placeholder row

    print("Final grid centers:", grid_centers)  # Debug: print final grid





    return grid_centers, True

def draw_missing_squares(image, grid_centers):
    """
    Draw red circles for missing squares based on the interpolated grid using data from all rows and columns.
    Also, draw a blue line connecting the centers of all detected squares.
    Args:
        image: Input image.
        grid_centers: List of grid centers with None for missing squares.
    """
    # Collect all valid square centers (excluding None values)
    valid_centers = []
    for row in grid_centers:
        for center in row:
            if center is not None:
                valid_centers.append(center)

    # Calculate the average x and y coordinates from all valid centers

    rows = len(valid_centers)
    cols = len(valid_centers[0])

    for i in range(rows):
        for j in range(cols):
            if valid_centers[i][j] is None:
                neighbors = []
                # Check neighboring points (left, right, top, bottom)
                if i > 0 and valid_centers[i - 1][j] is not None:  # Above
                    neighbors.append(valid_centers[i - 1][j])
                if i < rows - 1 and valid_centers[i + 1][j] is not None:  # Below
                    neighbors.append(valid_centers[i + 1][j])
                if j > 0 and valid_centers[i][j - 1] is not None:  # Left
                    neighbors.append(valid_centers[i][j - 1])
                if j < cols - 1 and valid_centers[i][j + 1] is not None:  # Right
                    neighbors.append(valid_centers[i][j + 1])

                # If there are valid neighbors, calculate the average
                if neighbors:
                    print("Neighbours Found for Missing Squares: "+neighbors)
                    avg_x = np.mean([coord[0] for coord in neighbors])
                    avg_y = np.mean([coord[1] for coord in neighbors])
                    valid_centers[i][j] = (avg_x, avg_y)

    print("Final grid centers after calculation:", valid_centers)



    # Draw blue line connecting all valid square centers
    for i in range(len(valid_centers) - 1):
        pt1 = valid_centers[i]
        pt2 = valid_centers[i + 1]
        cv2.line(image, pt1, pt2, (255, 0, 0), 10)  # Blue line, thickness 2



def detect_and_validate_partial_grid(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Resize image for consistent processing
    image = imutils.resize(image, width=800)
    cv2.imshow("Original Image", image)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (9, 9), 0)

    # Perform Edge Detection
    edges = cv2.Canny(blurred, 100, 200)
    cv2.imshow("Edges", edges)  # Debug: visualize edges

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Visualize all contours for debugging
    debug_image = image.copy()
    for contour in contours:
        cv2.drawContours(debug_image, [contour], -1, (255, 0, 0), 2)  # Blue for all contours
    cv2.imshow("All Contours", debug_image)

    # Loop through each contour and filter for squares
    squares = []
    for contour in contours:
        # Approximate the contour to reduce the number of vertices
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour has 4 vertices and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # Calculate aspect ratio to ensure it's a square
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Consider it a square if aspect ratio is approximately 1
            if 0.9 <= aspect_ratio <= 1.1 and cv2.contourArea(contour) > 100:
                squares.append(approx)

    # Debug: visualize detected squares
    debug_image = image.copy()
    for square in squares:
        cv2.drawContours(debug_image, [square], -1, (0, 255, 255), 2)  # Yellow for valid squares
    cv2.imshow("Filtered Squares", debug_image)

    # Validate approximate 3x3 grid
    grid_centers, success = is_approx_3x3_layout(squares)
    if success:
        print("Approximate 3x3 Layout detected!")
        # Draw detected squares
        for square in squares:
            cv2.drawContours(image, [square], -1, (0, 255, 0), 3)

        # Mark missing squares with red circles
        draw_missing_squares(image, grid_centers)
    else:
        print("3x3 Layout not detected.")

    # Display the results
    cv2.imshow("Detected Squares and Grid", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'image.png' with the path to your uploaded image
detect_and_validate_partial_grid("test4.jpg")