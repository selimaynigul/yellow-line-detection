import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_dynamic_hsv_range(hsv_image, target_hue_range=(15, 35), min_saturation=30, min_value=100):
    """
    Dynamically calculates the HSV range for yellow tones in the given image.
    """
    h, s, v = custom_split(hsv_image)
    
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


def custom_split(hsv_image):
    """
    Splits an HSV image into its Hue, Saturation, and Value channels.

    Parameters:
        hsv_image (numpy.ndarray): A 3D numpy array representing an HSV image
                                   with dimensions (height, width, 3).

    Returns:
        tuple: Three 2D numpy arrays (H, S, V) representing the individual channels.
    """
    # Ensure the input is a valid HSV image
    if len(hsv_image.shape) != 3 or hsv_image.shape[2] != 3:
        raise ValueError("Input must be an HSV image with 3 channels.")
    
    # Extract channels
    H = hsv_image[:, :, 0]
    S = hsv_image[:, :, 1]
    V = hsv_image[:, :, 2]
    
    return H, S, V


def apply_morphology(mask):
    """
    Applies enhanced morphological operations to clean the mask.
    """
    # Define a larger kernel for morphological operations
    kernel = custom_get_structuring_element('rect', (7, 7))  # Larger kernel for stronger effect

    # Apply multiple iterations of opening to remove noise
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Increase iterations

    # Apply multiple iterations of closing to fill gaps
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)  # Increase iterations

    return mask_cleaned

def custom_get_structuring_element(shape, ksize):
    """
    Creates a structuring element (kernel) for morphological operations.

    Parameters:
        shape (str): Shape of the structuring element. Currently supports 'rect'.
        ksize (tuple): Size of the structuring element (height, width).

    Returns:
        numpy.ndarray: A binary structuring element of the specified shape and size.
    """
    if shape == 'rect':
        # Create a rectangular kernel of the specified size
        return np.ones(ksize, dtype=np.uint8)
    else:
        raise ValueError("Unsupported shape. Currently only 'rect' is implemented.")
    
    
def custom_morph_open(image, kernel, iterations=1):
    """
    Perform morphological opening (erosion followed by dilation) on a binary image.

    Parameters:
        image (numpy.ndarray): Binary input image (2D array).
        kernel (numpy.ndarray): Structuring element (2D array).
        iterations (int): Number of times to apply the opening operation.
    
    Returns:
        numpy.ndarray: Binary image after opening.
    """
    result = image
    for _ in range(iterations):
        result = custom_erosion(result, kernel)
        result = custom_dilation(result, kernel)
    return result

def custom_morph_close(image, kernel, iterations=1):
    """
    Perform morphological closing (dilation followed by erosion) on a binary image.
    
    Parameters:
        image (numpy.ndarray): Binary input image (2D array).
        kernel (numpy.ndarray): Structuring element (2D array).
        iterations (int): Number of times to apply the closing operation.
    
    Returns:
        numpy.ndarray: Binary image after closing.
    """
    result = image
    for _ in range(iterations):
        result = custom_dilation(result, kernel)
        result = custom_erosion(result, kernel)
    return result


def custom_erosion(image, kernel):
    """
    Perform erosion on a binary image using a custom structuring element.
    
    Parameters:
        image (numpy.ndarray): Binary input image (2D array).
        kernel (numpy.ndarray): Structuring element (2D array).
    
    Returns:
        numpy.ndarray: Eroded binary image.
    """
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2
    
    # Pad the image to handle border pixels
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.uint8)
    
    # Slide the kernel across the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the region of interest
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            # Check if the kernel fits completely
            if np.all(region[kernel == 1] == 1):
                output[i, j] = 1  # Set to 1 if kernel fits
    
    return output

def custom_dilation(image, kernel):
    """
    Perform dilation on a binary image using a custom structuring element.
    
    Parameters:
        image (numpy.ndarray): Binary input image (2D array).
        kernel (numpy.ndarray): Structuring element (2D array).
    
    Returns:
        numpy.ndarray: Dilated binary image.
    """
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2
    
    # Pad the image to handle border pixels
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.uint8)
    
    # Slide the kernel across the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the region of interest
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            # Check if any part of the kernel matches the region
            if np.any(region[kernel == 1] == 1):
                output[i, j] = 1  # Set to 1 if any overlap occurs
    
    return output


    
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

# List of image filenames
image_files = ["cc.png", "dd.jpg", "image.png"]

# Process each image and store results
results = [process_image(image_file) for image_file in image_files]

# Display results for all images
plt.figure(figsize=(20, 20))
num_images = len(image_files)
for i, (image_file, (original, mask, cleaned_mask, highlight)) in enumerate(zip(image_files, results)):
    # Original Image
    plt.subplot(num_images, 4, i * 4 + 1)
    plt.title(f"Original Image ({image_file})")
    plt.imshow(original)
    plt.axis("off")

    # Original Mask
    plt.subplot(num_images, 4, i * 4 + 2)
    plt.title(f"Original Yellow Mask ({image_file})")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    # Cleaned Mask
    plt.subplot(num_images, 4, i * 4 + 3)
    plt.title(f"Cleaned Yellow Mask ({image_file})")
    plt.imshow(cleaned_mask, cmap="gray")
    plt.axis("off")

    # Semi-Transparent Highlight
    plt.subplot(num_images, 4, i * 4 + 4)
    plt.title(f"Semi-Transparent Highlight ({image_file})")
    plt.imshow(highlight)
    plt.axis("off")

plt.tight_layout()
plt.show()
