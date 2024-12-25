import numpy as np
import cv2
import matplotlib.pyplot as plt

def custom_rgb_to_hsv(image):
    """
    Converts an RGB image to HSV format using only NumPy.

    Parameters:
        image (numpy.ndarray): The input image in RGB format (H x W x 3).

    Returns:
        numpy.ndarray: The image converted to HSV format (H x W x 3), where:
                       H in [0, 360),
                       S in [0, 1],
                       V in [0, 1].
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel RGB image (H x W x 3).")

    # Convert to float in [0, 1]
    image_float = image.astype('float32') / 255.0

    R = image_float[:, :, 0]
    G = image_float[:, :, 1]
    B = image_float[:, :, 2]

    # Value channel (max of R, G, B)
    V = np.max(image_float, axis=2)

    # Chroma = max(R, G, B) - min(R, G, B)
    C = V - np.min(image_float, axis=2)

    # Saturation
    S = np.zeros_like(V)
    nonzero_mask = (V != 0)
    S[nonzero_mask] = C[nonzero_mask] / V[nonzero_mask]

    # Hue
    H = np.zeros_like(V)
    C_nonzero = (C != 0)  # Avoid division by zero

    # For pixels where V == R
    mask_r = (V == R) & C_nonzero
    H[mask_r] = (60.0 * ((G[mask_r] - B[mask_r]) / C[mask_r])) % 360

    # For pixels where V == G
    mask_g = (V == G) & C_nonzero
    H[mask_g] = (60.0 * ((B[mask_g] - R[mask_g]) / C[mask_g]) + 120) % 360

    # For pixels where V == B
    mask_b = (V == B) & C_nonzero
    H[mask_b] = (60.0 * ((R[mask_b] - G[mask_b]) / C[mask_b]) + 240) % 360

    # Combine H, S, and V into one HSV image
    hsv_image = np.stack([H, S, V], axis=-1)

    return hsv_image

if __name__ == "__main__":
    # Load test image from file
    test_image_path = "images/cc.png"
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        raise FileNotFoundError(f"Test image not found at path: {test_image_path}")

    # Convert BGR to RGB
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

    # Convert to HSV
   # hsv_result = custom_rgb_to_hsv(test_image)
    hsv_result = cv2.cvtColor(test_image, cv2.COLOR_RGB2HSV)

    # Visualize results
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original RGB Image")
    plt.imshow(test_image)
    plt.axis("off")
    
    # HSV result (display only Hue for visualization)
    plt.subplot(1, 2, 2)
    plt.title("Custom HSV (Hue Component)")
    plt.imshow(hsv_result[..., 0], cmap="hsv")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
