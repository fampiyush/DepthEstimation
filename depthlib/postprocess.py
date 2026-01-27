"""
Post-processing utilities for disparity maps.
"""

from typing import Optional

import cv2
import numpy as np

def filter_speckles(disparity, max_speckle_size=100, max_diff=1):
    """
    Remove small isolated regions (speckles) from disparity map.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    max_speckle_size : int
        Maximum size of speckle region to filter (pixels)
    max_diff : float
        Maximum disparity difference to consider as same region
    
    Returns:
    --------
    filtered : np.ndarray
        Filtered disparity map
    """
    filtered = disparity.copy()
    
    # Convert to 16-bit fixed-point for OpenCV
    disp_16s = (filtered * 16.0).astype(np.int16)
    
    # Filter speckles
    cv2.filterSpeckles(disp_16s, 0, max_speckle_size, int(max_diff * 16))
    
    # Convert back to float32
    filtered = disp_16s.astype(np.float32) / 16.0
    
    return filtered

def detect_outliers(disparity, threshold=3.0, kernel_size=5):
    """
    Detect outliers in disparity map using local statistics.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    threshold : float
        Number of standard deviations for outlier detection
    kernel_size : int
        Size of local neighborhood for statistics
        
    Returns:
    --------
    mask : np.ndarray (bool)
        Boolean mask where True indicates outliers
    """
    # Create a mask for valid disparities
    valid_mask = disparity > 0
    
    # Compute local mean and std using box filter
    mean = cv2.boxFilter(disparity, -1, (kernel_size, kernel_size))
    
    # Compute local standard deviation
    disparity_sq = disparity ** 2
    mean_sq = cv2.boxFilter(disparity_sq, -1, (kernel_size, kernel_size))
    std = np.sqrt(np.maximum(mean_sq - mean ** 2, 0))
    
    # Detect outliers
    diff = np.abs(disparity - mean)
    outlier_mask = (diff > threshold * std) & valid_mask
    
    return outlier_mask

def fill_holes(disparity, mask=None, method='inpaint', kernel_size=5):
    """
    Fill holes and invalid regions in disparity map.
    
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    mask : np.ndarray (bool), optional
        Boolean mask indicating holes to fill (True = hole)
        If None, fills all zero/invalid values
    method : str
        Filling method: 'inpaint' or 'nearest'
    kernel_size : int
        Kernel size for morphological operations
        
    Returns:
    --------
    filled : np.ndarray
        Disparity map with filled holes
    """
    filled = disparity.copy()
    
    # Create hole mask if not provided
    if mask is None:
        mask = (disparity <= 0)
    
    # Convert mask to uint8 for OpenCV
    hole_mask = mask.astype(np.uint8) * 255
    
    if method == 'inpaint':
        # Use Telea or NS inpainting algorithm
        filled = cv2.inpaint(filled.astype(np.float32), hole_mask, 
                            kernel_size, cv2.INPAINT_TELEA)
    elif method == 'nearest':
        # Dilate valid regions to fill small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                          (kernel_size, kernel_size))
        # Create distance transform from valid pixels
        dist = cv2.distanceTransform((~mask).astype(np.uint8), 
                                     cv2.DIST_L2, 5)
        # Fill with nearest valid neighbor
        for _ in range(kernel_size):
            dilated = cv2.dilate(filled, kernel)
            filled = np.where(mask, dilated, filled)
    
    return filled

def postprocess_disparity(disparity, **kwargs):
    """
    Apply a series of post-processing steps to refine the disparity map.
    Parameters:
    -----------
    disparity : np.ndarray
        Input disparity map
    **kwargs : dict
        Additional parameters for post-processing steps:
        - max_speckle_size: int (default: 50)
        - max_diff: float (default: 1)
        - outlier_threshold: float (default: 3.0)
        - outlier_kernel: int (default: 5)
        - fill_method: str (default: 'inpaint')
        - fill_kernel: int (default: 5)
        - apply_outlier_removal: bool (default: True)
        - apply_hole_filling: bool (default: True)
    
    Returns:
    --------
    refined_disparity : np.ndarray
    """
        
    # Step 1: Quick speckle removal
    result = filter_speckles(
        disparity.copy(),
        kwargs.get('max_speckle_size', 50),
        kwargs.get('max_diff', 1)
    )
    
    # Step 2: Outlier detection and masking
    if kwargs.get('apply_outlier_removal', True):
        outlier_mask = detect_outliers(
            result,
            threshold=kwargs.get('outlier_threshold', 3.0),
            kernel_size=kwargs.get('outlier_kernel', 5)
        )
        # Set outliers to zero (invalid)
        result[outlier_mask] = 0
    
    # Step 3: Hole filling
    if kwargs.get('apply_hole_filling', True):
        result = fill_holes(
            result,
            method=kwargs.get('fill_method', 'inpaint'),
            kernel_size=kwargs.get('fill_kernel', 3)
        )
    
    # Step 4: Fast 3x3 median filter
    output = cv2.medianBlur(result.astype(np.float32), 3)

    return output
