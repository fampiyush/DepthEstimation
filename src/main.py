<<<<<<< Updated upstream
from SimpleSGBM import run_sgbm_stage, build_sgbm
from input import load_stereo_pair
from visualizations import (visualize_stereo_pair, visualize_disparity, 
                            visualize_depth, visualize_disparity_and_depth)
from rectify import _load_calibration_cached
from postprocess import postprocess_disparity, enhance_disparity_map
import cv2
import numpy as np
=======
from input import load_stereo_pair, visualize_stereo_pair
>>>>>>> Stashed changes

if __name__ == "__main__":
    left_image_path = '../assets/stereo_pairs/im0.png'
    right_image_path = '../assets/stereo_pairs/im1.png'

<<<<<<< Updated upstream
    # Load original stereo pair
    left_img_rgb, right_img_rgb = load_stereo_pair(left_image_path, right_image_path)
    visualize_stereo_pair(left_img_rgb, right_img_rgb, 
                         title_left='Left Image (Original)', 
                         title_right='Right Image (Original)')

    # Check if images are already rectified (Middlebury datasets are pre-rectified)
    # If images are already rectified, we can skip the rectification step
    # and just convert to grayscale for SGBM
    
    # Load calibration data
    calib_data_dict = _load_calibration_cached('../assets/calib.txt')
    
    # Convert to grayscale directly (images already rectified)
    left_gray = cv2.cvtColor(left_img_rgb, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_img_rgb, cv2.COLOR_RGB2GRAY)
    
    visualize_stereo_pair(left_gray, right_gray,
                         title_left='Left Image (Grayscale)',
                         title_right='Right Image (Grayscale)')

    ndisp = 280  # From calib.txt
    sgbm = build_sgbm(num_disp=ndisp)
    
    # Extract calibration parameters
    focal_length = float(calib_data_dict["cam0"][0, 0])  # fx from camera matrix
    baseline_mm = float(calib_data_dict["baseline"])
    doffs = float(calib_data_dict.get("doffs", 0.0))
    
    disparity_px, depth_m = run_sgbm_stage(
        left_gray, right_gray,
        f_pixels=focal_length,
        baseline_m=baseline_mm / 1000.0,  # Convert mm to m
        doffs=doffs,  # Disparity offset
        min_disparity=0.5,  # Minimum valid disparity (adjustable)
        max_depth=50.0,  # Maximum depth to display (adjustable based on scene)
        sgbm=sgbm
    )
    
    # Print raw disparity statistics
    valid_disp = disparity_px > 0
    print("\n=== Raw Disparity Statistics ===")
    print(f"Disparity range: {disparity_px[valid_disp].min():.2f} - {disparity_px[valid_disp].max():.2f} pixels")
    print(f"Invalid disparities: {(~valid_disp).sum() / valid_disp.size * 100:.1f}%")
    
    # Apply post-processing to reduce noise
    print("\n=== Applying Post-Processing ===")
    print("1. Filtering speckles...")
    disparity_filtered = postprocess_disparity(
        disparity_px, 
        left_image=left_gray,
        method='all',  # Apply all filters: speckle, bilateral, median, fill
        max_speckle_size=100,  # Remove isolated regions smaller than this
        max_diff=1.0,           # Speckle filter disparity threshold
        d=9,                    # Bilateral filter neighborhood
        sigma_color=75,         # Bilateral filter color sigma
        sigma_space=75,         # Bilateral filter space sigma
        kernel_size=5,          # Median filter kernel
        max_hole_size=10        # Fill holes smaller than this
    )
    
    # Enhance disparity for better visualization
    disparity_enhanced = enhance_disparity_map(disparity_filtered, clip_percentile=1)
    
    # Recompute depth from filtered disparity
    print("2. Recomputing depth from filtered disparity...")
    from SimpleSGBM import disparity_to_depth
    depth_filtered = disparity_to_depth(
        disparity_filtered, 
        focal_length, 
        baseline_mm / 1000.0, 
        doffs,
        eps=0.5,
        max_depth=50.0
    )
    
    # Print filtered statistics
    valid_disp_filtered = disparity_filtered > 0
    print(f"\nFiltered disparity range: {disparity_filtered[valid_disp_filtered].min():.2f} - {disparity_filtered[valid_disp_filtered].max():.2f} pixels")
    print(f"Invalid disparities after filtering: {(~valid_disp_filtered).sum() / valid_disp_filtered.size * 100:.1f}%")
    
    if depth_filtered is not None:
        valid_depth = np.isfinite(depth_filtered) & (depth_filtered > 0)
        if valid_depth.any():
            print(f"Filtered depth range: {depth_filtered[valid_depth].min():.2f} - {depth_filtered[valid_depth].max():.2f} meters")
            print(f"Invalid depth values: {(~valid_depth).sum() / valid_depth.size * 100:.1f}%")
    
    # Visualizations
    print("\n=== Generating Visualizations ===")
    
    # Compare raw vs filtered disparity
    print("Showing raw disparity map...")
    visualize_disparity(disparity_px, title='Disparity Map (Raw)', cmap='jet')
    
    print("Showing filtered disparity map...")
    visualize_disparity(disparity_enhanced, title='Disparity Map (Filtered & Enhanced)', cmap='jet')
    
    # Compare raw vs filtered depth
    print("Showing raw depth map...")
    visualize_depth(depth_m, title='Depth Map (Raw)', cmap='turbo_r')
    
    print("Showing filtered depth map...")
    visualize_depth(depth_filtered, title='Depth Map (Filtered)', cmap='turbo_r')
    
    # Final combined visualization with filtered results
    print("Showing combined visualization...")
    visualize_disparity_and_depth(disparity_enhanced, depth_filtered, left_img=left_gray)
=======
    left_img_rgb, right_img_rgb = load_stereo_pair(left_image_path, right_image_path)
    visualize_stereo_pair(left_img_rgb, right_img_rgb)
>>>>>>> Stashed changes
