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

# Step 5: Highlight Yellow Regions in the Original Image
def highlight_yellow_regions(image, mask):
    # Convert mask to a 3-channel image
    mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Highlight yellow regions by combining the original image with the mask
    highlighted_image = cv2.bitwise_and(image, mask_3_channel)
    return highlighted_image

# Step 6: Display or Save the Result
def display_result(original_image, mask, highlighted_image):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 2)
    plt.title("Yellow Mask")
    plt.imshow(mask, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Highlighted Yellow Regions")
    plt.imshow(cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB))
    
    plt.show()

# Main Function to Run All Steps
def main(image_path):
    # Load and process the image
    image = load_image(image_path)
    hsv_image = convert_to_hsv(image)
    yellow_mask = create_yellow_mask(hsv_image)
    cleaned_mask = clean_mask(yellow_mask)
    highlighted_image = highlight_yellow_regions(image, cleaned_mask)
    
    # Display the results
    display_result(image, cleaned_mask, highlighted_image)

# Provide the path to your input image
if __name__ == "__main__":
    image_path = "image.jpg"  # Change to the path of your image
    main(image_path)
