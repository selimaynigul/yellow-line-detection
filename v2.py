import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# Manual Canny Edge Detection
def manual_canny(image, low_threshold, high_threshold):
    # Convert to grayscale
    gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

    # Apply Gaussian Blur
    blurred = gaussian_filter(gray, sigma=1.4)

    # Compute Gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad_x = convolve(blurred, sobel_x, mode='reflect')
    grad_y = convolve(blurred, sobel_y, mode='reflect')
    
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi) % 180

    # Non-Maximum Suppression
    thin_edges = non_max_suppression(gradient_magnitude, gradient_direction)

    # Double Threshold and Edge Tracking
    edges = double_threshold(thin_edges, low_threshold, high_threshold)

    return edges

def non_max_suppression(gradient_magnitude, gradient_direction):
    rows, cols = gradient_magnitude.shape
    result = np.zeros((rows, cols), dtype=np.float32)
    angle = gradient_direction

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            try:
                # Determine pixel neighbors to interpolate
                q = 255
                r = 255

                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = gradient_magnitude[i, j + 1]
                    r = gradient_magnitude[i, j - 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    q = gradient_magnitude[i + 1, j - 1]
                    r = gradient_magnitude[i - 1, j + 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    q = gradient_magnitude[i + 1, j]
                    r = gradient_magnitude[i - 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    q = gradient_magnitude[i - 1, j - 1]
                    r = gradient_magnitude[i + 1, j + 1]

                if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                    result[i, j] = gradient_magnitude[i, j]
                else:
                    result[i, j] = 0

            except IndexError as e:
                pass

    return result

def double_threshold(image, low_threshold, high_threshold):
    strong = 255
    weak = 75

    result = np.zeros_like(image, dtype=np.uint8)
    strong_i, strong_j = np.where(image >= high_threshold)
    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    return result

# Manual Hough Line Transform
def manual_hough_lines(edge_image, rho_res=1, theta_res=np.pi/180, threshold=100):
    rows, cols = edge_image.shape
    max_rho = int(np.sqrt(rows**2 + cols**2))
    accumulator = np.zeros((2 * max_rho, int(np.pi / theta_res)))

    # Step 1: Accumulator Voting
    edge_points = np.argwhere(edge_image)
    for y, x in edge_points:
        for theta_index, theta in enumerate(np.arange(0, np.pi, theta_res)):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            accumulator[rho + max_rho, theta_index] += 1

    # Step 2: Detect Peaks
    lines = []
    for rho_index, theta_index in np.argwhere(accumulator > threshold):
        rho = rho_index - max_rho
        theta = theta_index * theta_res
        lines.append((rho, theta))

    return lines

# Draw Detected Lines
def draw_lines(image, lines):
    draw = ImageDraw.Draw(image)
    for rho, theta in lines:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=2)
    return image

# Main Workflow
if __name__ == '__main__':
    # Load image
    image = Image.open('image.jpg').convert('RGB')
    image_array = np.array(image)

    # Detect edges
    edges = manual_canny(image_array, low_threshold=50, high_threshold=150)

    # Detect lines
    lines = manual_hough_lines(edges, rho_res=1, theta_res=np.pi/180, threshold=100)

    # Draw lines
    output_image = draw_lines(image.copy(), lines)

    # Display results
    plt.imshow(output_image)
    plt.axis('off')
    plt.show()
