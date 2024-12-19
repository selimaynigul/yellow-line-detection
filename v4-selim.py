import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_dynamic_hsv_range(hsv_image, target_hue_range=(15, 35), min_saturation=30, min_value=100):
    """
    Dynamically calculates the HSV range for yellow tones in the given image.
    """
    h, s, v = cv2.split(hsv_image)
    
    # Mask for the target hue range
    hue_mask = (h >= target_hue_range[0]) & (h <= target_hue_range[1])
    
    # Mask for sufficient saturation and brightness
    valid_mask = hue_mask & (s >= min_saturation) & (v >= min_value)
    
    # Extract hue values within the mask
    valid_hues = h[valid_mask]
    
    # If valid hues are found, calculate dynamic range
    if len(valid_hues) > 0:
        lower_hue = max(target_hue_range[0], np.percentile(valid_hues, 2))  # 2nd percentile
        upper_hue = min(target_hue_range[1], np.percentile(valid_hues, 98))  # 98th percentile
    else:
        # Default to full range if no yellow found
        lower_hue, upper_hue = target_hue_range[0], target_hue_range[1]
    
    return np.array([lower_hue, min_saturation, min_value]), np.array([upper_hue, 255, 255])

def apply_morphology(mask):
    """
    Applies enhanced morphological operations to clean the mask.
    """
    # Define a larger kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))  # Larger kernel for stronger effect

    # Apply multiple iterations of opening to remove noise
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Increase iterations

    # Apply multiple iterations of closing to fill gaps
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)  # Increase iterations

    return mask_cleaned

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistent display

    # Preprocess the image (optional)
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0)  # Apply Gaussian blur to reduce noise

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2HSV)

    # Dynamically determine the HSV range for yellow
    lower_yellow, upper_yellow = get_dynamic_hsv_range(hsv_image)

    # Convert to np.uint8 to ensure compatibility with cv2.inRange
    lower_yellow = lower_yellow.astype(np.uint8)
    upper_yellow = upper_yellow.astype(np.uint8)

    print(f"Dynamic HSV Range for Yellow in {image_path}: Lower {lower_yellow}, Upper {upper_yellow}")

    # Create a mask for yellow
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # Apply morphological operations to clean the mask
    yellow_mask_cleaned = apply_morphology(yellow_mask)

    # Apply the cleaned mask to highlight the yellow areas
    highlighted = image.copy()
    highlighted[yellow_mask_cleaned > 0] = [255, 0, 0]  # Highlight in red

    # Create a semi-transparent overlay
    overlay = image.copy()
    overlay[yellow_mask_cleaned > 0] = [255, 0, 0]  # Highlight in red
    semi_transparent = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    return image, yellow_mask, yellow_mask_cleaned, semi_transparent

# Process both images
image1, mask1, cleaned_mask1, highlight1 = process_image("cc.png")
image2, mask2, cleaned_mask2, highlight2 = process_image("bb.jpg")

# Display results for both images
plt.figure(figsize=(20, 15))

# Results for the first image
plt.subplot(4, 4, 1)
plt.title("Original Image (cc.png)")
plt.imshow(image1)
plt.axis("off")

plt.subplot(4, 4, 2)
plt.title("Original Yellow Mask (cc.png)")
plt.imshow(mask1, cmap="gray")
plt.axis("off")

plt.subplot(4, 4, 3)
plt.title("Cleaned Yellow Mask (cc.png)")
plt.imshow(cleaned_mask1, cmap="gray")
plt.axis("off")

plt.subplot(4, 4, 4)
plt.title("Semi-Transparent Highlight (cc.png)")
plt.imshow(highlight1)
plt.axis("off")

# Results for the second image
plt.subplot(4, 4, 5)
plt.title("Original Image (bb.jpg)")
plt.imshow(image2)
plt.axis("off")

plt.subplot(4, 4, 6)
plt.title("Original Yellow Mask (bb.jpg)")
plt.imshow(mask2, cmap="gray")
plt.axis("off")

plt.subplot(4, 4, 7)
plt.title("Cleaned Yellow Mask (bb.jpg)")
plt.imshow(cleaned_mask2, cmap="gray")
plt.axis("off")

plt.subplot(4, 4, 8)
plt.title("Semi-Transparent Highlight (bb.jpg)")
plt.imshow(highlight2)
plt.axis("off")

plt.tight_layout()
plt.show()
