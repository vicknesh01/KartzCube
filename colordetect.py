# This module is used to detect the color and save the values
# Created by Vickneshwaran Elangkeeran

import cv2
import imutils

print("KartzCube Color Detector. Powered by OpenCV " + cv2.__version__)



def is_approx_3x3_layout(squares, tolerance=30):
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

    # Check if we can interpolate a 3x3 grid
    if len(grid_centers) == 3:
        return grid_centers, True
    return [], False

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

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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

    # Validate approximate 3x3 grid
    grid_centers, success = is_approx_3x3_layout(squares)
    if success:
        print("Approximate 3x3 Layout detected!")
        # Draw detected squares
        for square in squares:
            cv2.drawContours(image, [square], -1, (0, 255, 0), 3)

        # Draw interpolated grid centers (red circles for missing squares)
        for row in grid_centers:
            for center in row:
                if center:
                    cv2.circle(image, center, 5, (0, 255, 0), -1)  # Green for detected squares
                else:
                    # Estimate missing squares' positions based on neighbors
                    missing_x = sum(c[0] for c in row if c) // len([c for c in row if c])
                    missing_y = sum(c[1] for c in row if c) // len([c for c in row if c])
                    cv2.circle(image, (missing_x, missing_y), 5, (0, 0, 255), -1)  # Red for missing
    else:
        print("3x3 Layout not detected.")

    # Display the results
    cv2.imshow("Detected Squares and Grid", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Replace 'test.jpg' with the path to your image
detect_and_validate_partial_grid("test.jpg")
