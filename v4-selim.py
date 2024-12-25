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
    #mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # Increase iterations
    mask_cleaned = custom_morphology_open(mask, kernel, iterations=2) # dogru calisiyor ama yavas


    # Apply multiple iterations of closing to fill gaps
    #mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)  # Increase iterations
    mask_cleaned = custom_morphology_close(mask, kernel, iterations=2) # biraz kotu calisiyor


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
    

def custom_gaussian_blur(image, kernel_size=(5, 5), sigma=0):
    """
    Applies a Gaussian blur to an image using only NumPy (no external libraries like OpenCV or SciPy).

    Parameters:
        image (numpy.ndarray): Input image (H x W x C) or (H x W) in RGB/Gray format.
        kernel_size (tuple): The size of the Gaussian kernel, e.g. (5, 5).
        sigma (float): The standard deviation of the Gaussian. If set to 0, it will be estimated
                       from the kernel size similar to how OpenCV does by default.

    Returns:
        numpy.ndarray: Blurred image of the same shape as the input.
    """
    # If sigma=0, estimate using a common heuristic:
    # OpenCV uses sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8 for each dimension, but we can simplify:
    if sigma <= 0:
        sigma = 0.3 * ((kernel_size[0] - 1) * 0.5 - 1) + 0.8

    # 1. Create the Gaussian kernel
    kernel = _create_gaussian_kernel(kernel_size, sigma)
    
    # 2. Convolve the image with the kernel
    #    Handle both color (H x W x C) and grayscale (H x W) images
    if len(image.shape) == 2:
        # Grayscale image
        blurred = _convolve2d(image, kernel)
    else:
        # Color image
        # Convolve each channel separately
        blurred_channels = []
        for c in range(image.shape[2]):
            channel_blurred = _convolve2d(image[..., c], kernel)
            blurred_channels.append(channel_blurred)
        blurred = np.stack(blurred_channels, axis=-1)

    # Convert to the same type as the input (important if you're expecting uint8)
    # Clip to [0, 255] if you want an 8-bit result. Otherwise you may just return as float.
    blurred = np.clip(blurred, 0, 255).astype(image.dtype)

    return blurred


def _create_gaussian_kernel(kernel_size, sigma):
    """
    Creates a 2D Gaussian kernel using the given kernel size and standard deviation (sigma).

    Parameters:
        kernel_size (tuple): (height, width) of the kernel, e.g. (5, 5)
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: A 2D Gaussian kernel of shape (kernel_size[0], kernel_size[1]).
    """
    kx, ky = kernel_size
    # Coordinates of the kernel center
    cx, cy = (kx - 1) / 2.0, (ky - 1) / 2.0

    # Initialize kernel
    kernel = np.zeros((kx, ky), dtype=np.float32)

    # Compute Gaussian for each cell (x, y)
    for x in range(kx):
        for y in range(ky):
            # Distance from center
            dx = (x - cx)**2
            dy = (y - cy)**2
            # 2D Gaussian formula
            kernel[x, y] = np.exp(-(dx + dy) / (2 * sigma**2))

    # Normalize so that the sum of all elements is 1
    kernel_sum = np.sum(kernel)
    if kernel_sum != 0:
        kernel /= kernel_sum

    return kernel


def _convolve2d(image, kernel):
    """
    Convolves a 2D image (single channel) with a 2D kernel using 'same' padding.

    Parameters:
        image (numpy.ndarray): Grayscale image of shape (H, W).
        kernel (numpy.ndarray): 2D kernel of shape (kH, kW).

    Returns:
        numpy.ndarray: Convolved image of shape (H, W).
    """
    # Image dimensions
    H, W = image.shape
    kH, kW = kernel.shape

    # Output array (float to avoid overflow if input is uint8)
    convolved = np.zeros((H, W), dtype=np.float32)

    # Amount of padding needed on each side
    pad_h = kH // 2
    pad_w = kW // 2

    # Pad the image with zeros (or reflect, replicate, etc. if desired)
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    # Perform the convolution
    for i in range(H):
        for j in range(W):
            # Extract the local region
            region = padded[i : i + kH, j : j + kW]

            # Element-wise multiply and accumulate
            value = np.sum(region * kernel)

            # Store in the output
            convolved[i, j] = value

    return convolved

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
        image (numpy.ndarray): Input RGB image (H x W x 3) with values in the range [0, 255].

    Returns:
        numpy.ndarray: HSV image (H x W x 3) with H in [0, 180] and S, V in [0, 255].
    """
    # Normalize RGB values to [0, 1]
    rgb_normalized = image.astype(np.float32) / 255.0

    # Separate channels
    R, G, B = rgb_normalized[..., 0], rgb_normalized[..., 1], rgb_normalized[..., 2]

    # Compute max and min values for each pixel
    max_val = np.max(rgb_normalized, axis=2)
    min_val = np.min(rgb_normalized, axis=2)
    delta = max_val - min_val

    # Initialize HSV arrays
    H = np.zeros_like(max_val)
    S = np.zeros_like(max_val)
    V = max_val

    # Calculate Hue
    mask_r = (max_val == R) & (delta > 0)
    mask_g = (max_val == G) & (delta > 0)
    mask_b = (max_val == B) & (delta > 0)

    # Assign hue values based on the dominant color channel
    H[mask_r] = ((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6
    H[mask_g] = ((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2
    H[mask_b] = ((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4

    # Convert hue to degrees
    H = (H / 6.0) * 180.0  # Scale to [0, 180] for compatibility

    # Calculate Saturation
    S[max_val > 0] = (delta[max_val > 0] / max_val[max_val > 0])

    # Scale S and V to [0, 255]
    S = (S * 255).astype(np.uint8)
    V = (V * 255).astype(np.uint8)

    # Stack H, S, and V to create the HSV image
    hsv_image = np.stack([H.astype(np.uint8), S, V], axis=2)

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

def process_video(input_video_path, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for output video
    
    # Initialize the video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit when no more frames are available

        # Convert the frame to RGB
        frame_rgb = custom_bgr_to_rgb(frame)
        
        # Preprocess and convert to HSV
        frame_blurred = custom_gaussian_blur(frame_rgb, (5, 5), 0)
        hsv_frame = custom_rgb_to_hsv(frame_blurred)
        
        # Dynamically determine the HSV range for yellow
        lower_yellow, upper_yellow = get_dynamic_hsv_range(hsv_frame)
        
        # Convert ranges to uint8 for compatibility
        lower_yellow = lower_yellow.astype(np.uint8)
        upper_yellow = upper_yellow.astype(np.uint8)
        
        # Create a mask for yellow
        yellow_mask = custom_in_range(hsv_frame, lower_yellow, upper_yellow)
        
        # Clean the mask with morphology
        yellow_mask_cleaned = apply_morphology(yellow_mask)
        
        # Highlight the yellow areas on the frame
        overlay = frame_rgb.copy()
        overlay[yellow_mask_cleaned > 0] = [255, 0, 0]  # Highlight in red
        semi_transparent = custom_add_weighted(frame_rgb, 0.7, overlay, 0.3, 0)
        
        # Convert back to BGR for video writer
        output_frame = cv2.cvtColor(semi_transparent, cv2.COLOR_RGB2BGR)
        
        # Write the processed frame to the output video
        out.write(output_frame)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved as {output_video_path}")

def boost_saturation(image, saturation_boost=50):
    """
    Boosts the saturation of an image while leaving other properties unchanged.
    
    Parameters:
        image (numpy.ndarray): Input RGB image (H x W x 3).
        saturation_boost (int): Amount to increase saturation (0-255).
        
    Returns:
        numpy.ndarray: RGB image with boosted saturation.
    """
    # Convert the image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Boost the saturation channel
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] + saturation_boost, 0, 255)
    
    # Convert back to RGB
    boosted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    
    return boosted_image


def refine_dynamic_hsv_range(hsv_image, target_hue_range=(15, 35), min_saturation=50, min_value=100):
    """
    Dynamically calculates a more refined HSV range for yellow tones in the given image.
    """
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    
    # Mask for the target hue range
    hue_mask = (h >= target_hue_range[0]) & (h <= target_hue_range[1])
    
    # Mask for sufficient saturation and brightness
    valid_mask = hue_mask & (s >= min_saturation) & (v >= min_value)
    
    # Extract hue values within the mask
    valid_hues = h[valid_mask]
    valid_saturations = s[valid_mask]
    valid_values = v[valid_mask]
    
    # If valid hues are found, calculate a more specific range
    if len(valid_hues) > 0:
        lower_hue = max(target_hue_range[0], np.percentile(valid_hues, 5))  # 5th percentile
        upper_hue = min(target_hue_range[1], np.percentile(valid_hues, 95))  # 95th percentile
        lower_saturation = max(min_saturation, np.percentile(valid_saturations, 5))
        upper_saturation = 255  # Keep full range
        lower_value = max(min_value, np.percentile(valid_values, 5))
        upper_value = 255  # Keep full range
    else:
        # Default to the provided ranges if no yellow is found
        lower_hue, upper_hue = target_hue_range[0], target_hue_range[1]
        lower_saturation, upper_saturation = min_saturation, 255
        lower_value, upper_value = min_value, 255
    
    return (
        np.array([lower_hue, lower_saturation, lower_value], dtype=np.uint8),
        np.array([upper_hue, upper_saturation, upper_value], dtype=np.uint8),
    )



def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    image =  custom_bgr_to_rgb(image) # Convert to RGB for consistent display

    # Preprocess the image (optional)
    #image_blurred = cv2.GaussianBlur(image, (5, 5), 0) 
    image_blurred = custom_gaussian_blur(image, (5, 5), 0)  # 5x5 Gaussian kernel, automatically determine sigma



    # Convert the image to HSV color space
    #hsv_image = cv2.cvtColor(image_blurred, cv2.COLOR_RGB2HSV)
    hsv_image = custom_rgb_to_hsv(image_blurred)  # fena degil is gorur


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

""" process_video("videos/video2.mp4", "outputs/output.mp4")
 """ 

# List of image filenames
image_files = ["images/cc.png","images/dd.jpg", "images/bb.jpg", "images/image.png"]
#image_files = ["images/cc.png"]

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
  