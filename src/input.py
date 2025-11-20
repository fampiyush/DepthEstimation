<<<<<<< Updated upstream
"""Input utilities for loading stereo image pairs."""

import cv2


def load_stereo_pair(left_image_path, right_image_path):
    """
    Load a stereo image pair from file paths.
    
    Parameters:
    -----------
    left_image_path : str
        Path to the left image
    right_image_path : str
        Path to the right image
    
    Returns:
    --------
    left_img_rgb : np.ndarray
        Left image in RGB format
    right_img_rgb : np.ndarray
        Right image in RGB format
    """
=======
import cv2
import matplotlib.pyplot as plt

def load_stereo_pair(left_image_path, right_image_path):
>>>>>>> Stashed changes
    # Load images
    left_img = cv2.imread(left_image_path)
    right_img = cv2.imread(right_image_path)

    if left_img is None or right_img is None:
        raise FileNotFoundError("One or both image paths are invalid.")

    # Convert from BGR (OpenCV default) to RGB for Matplotlib display
    left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

<<<<<<< Updated upstream
    return left_img_rgb, right_img_rgb
=======
    return left_img_rgb, right_img_rgb

def visualize_stereo_pair(left_img_rgb, right_img_rgb):
    # Display images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(left_img_rgb)
    axes[0].set_title('Left Image')
    axes[0].axis('off')

    axes[1].imshow(right_img_rgb)
    axes[1].set_title('Right Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
>>>>>>> Stashed changes
