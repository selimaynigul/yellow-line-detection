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
    #mask_cleaned = custom_morphology_open(mask, kernel, iterations=2) # dogru calisiyor ama yavas


    # Apply multiple iterations of closing to fill gaps
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)  # Increase iterations
    #mask_cleaned = custom_morphology_close(mask, kernel, iterations=2) # biraz kotu calisiyor


    return mask_cleaned


def custom_morphology_open(image, kernel, iterations=1):
    """
    Perform morphological opening (erosion followed by dilation) on a binary image.

    Parameters:
        image (np.ndarray): Input binary image (2D array).
        kernel (np.ndarray): Structuring element for morphological operations.
        iterations (int): Number of times to apply the opening operation.

    Returns:
        np.ndarray: Processed binary image after morphological opening.
    """
    def erode(image, kernel, iterations=1):
        for _ in range(iterations):
            padded_image = np.pad(image, kernel.shape[0] // 2, mode='constant', constant_values=0)
            eroded = np.zeros_like(image)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                    eroded[i, j] = np.min(region[kernel == 1])
            image = eroded
        return image

    def dilate(image, kernel, iterations=1):
        for _ in range(iterations):
            padded_image = np.pad(image, kernel.shape[0] // 2, mode='constant', constant_values=0)
            dilated = np.zeros_like(image)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                    dilated[i, j] = np.max(region[kernel == 1])
            image = dilated
        return image

    # Apply erosion followed by dilation
    eroded_image = erode(image, kernel, iterations=iterations)
    opened_image = dilate(eroded_image, kernel, iterations=iterations)
    return opened_image

def custom_morphology_close(image, kernel, iterations=1):
    """
    Perform morphological closing (dilation followed by erosion) on a binary image.

    Parameters:
        image (np.ndarray): Input binary image (2D array).
        kernel (np.ndarray): Structuring element for morphological operations.
        iterations (int): Number of times to apply the closing operation.

    Returns:
        np.ndarray: Processed binary image after morphological closing.
    """
    def erode(image, kernel, iterations=1):
        for _ in range(iterations):
            padded_image = np.pad(image, kernel.shape[0] // 2, mode='constant', constant_values=0)
            eroded = np.zeros_like(image)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                    eroded[i, j] = np.min(region[kernel == 1])
            image = eroded
        return image

    def dilate(image, kernel, iterations=1):
        for _ in range(iterations):
            padded_image = np.pad(image, kernel.shape[0] // 2, mode='constant', constant_values=0)
            dilated = np.zeros_like(image)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    region = padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                    dilated[i, j] = np.max(region[kernel == 1])
            image = dilated
        return image

    # Apply dilation followed by erosion
    dilated_image = dilate(image, kernel, iterations=iterations)
    closed_image = erode(dilated_image, kernel, iterations=iterations)
    return closed_image

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
    

def custom_bgr_to_rgb(image):
    """
    Converts a BGR image to RGB format.

    Parameters:
        image (numpy.ndarray): The input image in BGR format (H x W x 3).

    Returns:
        numpy.ndarray: The image converted to RGB format (H x W x 3).
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel image (H x W x 3).")
    
    # Swap the first (B) and third (R) channels
    rgb_image = image[:, :, ::-1]
    return rgb_image


def custom_rgb_to_hsv(image):
    """
    Converts an RGB image to HSV format.

    Parameters:
        image (numpy.ndarray): The input image in RGB format (H x W x 3).

    Returns:
        numpy.ndarray: The image converted to HSV format (H x W x 3).
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image (H x W x 3).")

    # Normalize RGB values to [0, 1] range
    image = image.astype('float32') / 255.0
    
    # Split RGB channels
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    # Compute Value (V)
    V = np.max(image, axis=2)
    
    # Compute Saturation (S)
    C = V - np.min(image, axis=2)  # Chroma
    S = np.where(V != 0, C / V, 0)
    
    # Compute Hue (H)
    H = np.zeros_like(V)
    mask = (V == R)
    H[mask] = (60 * ((G[mask] - B[mask]) / C[mask]) + 360) % 360
    mask = (V == G)
    H[mask] = (60 * ((B[mask] - R[mask]) / C[mask]) + 120) % 360
    mask = (V == B)
    H[mask] = (60 * ((R[mask] - G[mask]) / C[mask]) + 240) % 360
    
    # Set Hue to 0 where Chroma is 0
    H[C == 0] = 0

    # Combine H, S, V into an HSV image
    hsv_image = np.stack([H, S, V], axis=2)
    return hsv_image

def custom_in_range(image, lower_bound, upper_bound):
    """
    Creates a binary mask where each pixel is 1 if it is within the given range, and 0 otherwise.

    Parameters:
        image (numpy.ndarray): Input image (e.g., HSV image) of shape (H x W x C).
        lower_bound (tuple or list): Lower bound for the range (e.g., [H_min, S_min, V_min]).
        upper_bound (tuple or list): Upper bound for the range (e.g., [H_max, S_max, V_max]).

    Returns:
        numpy.ndarray: Binary mask of shape (H x W), where 1 indicates a pixel within range.
    """
    # Ensure the bounds are NumPy arrays for comparison
    lower_bound = np.array(lower_bound, dtype=np.float32)
    upper_bound = np.array(upper_bound, dtype=np.float32)
    
    # Create a mask for each channel
    within_lower = image >= lower_bound
    within_upper = image <= upper_bound

    # Combine all channels to create the final mask
    mask = np.all(within_lower & within_upper, axis=2)
    
    # Convert boolean mask to uint8 (0 or 255)
    return (mask * 255).astype(np.uint8)

def custom_add_weighted(image1, alpha, image2, beta, gamma):
    """
    Blends two images using a weighted sum.

    Parameters:
        image1 (numpy.ndarray): First input image.
        alpha (float): Weight of the first image.
        image2 (numpy.ndarray): Second input image.
        beta (float): Weight of the second image.
        gamma (float): Scalar added to each sum.

    Returns:
        numpy.ndarray: Blended image.
    """
    # Ensure both images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Both images must have the same shape.")
    
    # Perform weighted addition
    blended = alpha * image1 + beta * image2 + gamma
    
    # Clip values to valid range [0, 255] and convert to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    image =  custom_bgr_to_rgb(image) # Convert to RGB for consistent display

    # Preprocess the image (optional)
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0) 

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2HSV)


    # Dynamically determine the HSV range for yellow
    lower_yellow, upper_yellow = get_dynamic_hsv_range(hsv_image)

    # Convert to np.uint8 to ensure compatibility
    lower_yellow = lower_yellow.astype(np.uint8)
    upper_yellow = upper_yellow.astype(np.uint8)

    print(f"Dynamic HSV Range for Yellow in {image_path}: Lower {lower_yellow}, Upper {upper_yellow}")

    # Create a mask for yellow
    yellow_mask = custom_in_range(hsv_image, lower_yellow, upper_yellow)


    # Apply morphological operations to clean the mask
    yellow_mask_cleaned = apply_morphology(yellow_mask)

    # Apply the cleaned mask to highlight the yellow areas
    highlighted = image.copy()
    highlighted[yellow_mask_cleaned > 0] = [255, 0, 0]  # Highlight in red

    # Create a semi-transparent overlay
    overlay = image.copy()
    overlay[yellow_mask_cleaned > 0] = [255, 0, 0]  # Highlight in red
    semi_transparent = custom_add_weighted(image, 0.7, overlay, 0.3, 0)

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
