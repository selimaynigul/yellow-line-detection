import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the Image
def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# Step 2: Convert to HSV Color Space
def convert_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image

# Step 3: Create a Mask for Yellow Color
def create_yellow_mask(hsv_image):
    lower_yellow = np.array([20, 100, 100])  # Adjust based on lighting
    upper_yellow = np.array([30, 255, 255]) # Adjust based on lighting
    mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    return mask

# Step 4: Apply Morphological Operations
def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    return cleaned_mask

# Step 5: Edge Detection
def detect_edges(mask):
    edges = cv2.Canny(mask, 50, 150)
    return edges

# Step 6: Line Detection Using Hough Transform
def detect_lines(edges, image):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

# Step 7: Display or Save the Result
def display_result(image, mask, edges):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 2)
    plt.title("Yellow Mask")
    plt.imshow(mask, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Edges")
    plt.imshow(edges, cmap='gray')
    
    plt.show()

# Main Function to Run All Steps
def main(image_path):
    # Load and process the image
    image = load_image(image_path)
    hsv_image = convert_to_hsv(image)
    yellow_mask = create_yellow_mask(hsv_image)
    cleaned_mask = clean_mask(yellow_mask)
    edges = detect_edges(cleaned_mask)
    result_image = detect_lines(edges, image.copy())
    
    # Display the results
    display_result(result_image, cleaned_mask, edges)

# Provide the path to your input image
if __name__ == "__main__":
    image_path = "image.jpg"  # Change to the path of your image
    main(image_path)
