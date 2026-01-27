import numpy as np
import pytest
from depthlib.rectify import rectify_images

def test_rectification_output_shape():
    """
    Ensure rectification returns images of the exact requested size.
    """
    # 1. Create fake random images (480p)
    img_L = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_R = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 2. Fake Calibration (Identity matrices = no distortion)
    K = np.eye(3)
    D = np.zeros(5)
    R = np.eye(3)
    T = np.array([0.5, 0, 0]) # 50cm baseline
    
    # 3. Run Rectification
    rect_L, rect_R = rectify_images(
        img_L, img_R,
        cam_matrix_L=K, cam_matrix_R=K,
        dist_coeff_L=D, dist_coeff_R=D,
        baseline=0.5,
        image_width=640, image_height=480,
        rotation=R, translation=T
    )
    
    # 4. Assertions
    assert rect_L.shape == (480, 640) # Should be grayscale (2D)
    assert rect_R.shape == (480, 640)
    assert rect_L.dtype == np.uint8