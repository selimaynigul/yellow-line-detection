import cv2
import numpy as np

# Load image
image = cv2.imread('image.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Yellow color mask
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])
mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

# Apply edge detection
edges = cv2.Canny(mask, 50, 150)

# Detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

# Draw lines on the original image
output = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Show results
cv2.imshow('Yellow Lines', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
