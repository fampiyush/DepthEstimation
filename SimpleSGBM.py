import numpy as np
import cv2

"""
Pipeline Integration (Data Flow):
  Preprocess Module -> Rectify/Undistort -> [This Module: SGBM Disparity]
  -> Postprocess (e.g., WLS filter, left-right check)
  -> Depth Mapping (Z = f*B/d)
  -> Visualization / Point Cloud / Tracking

Note :
  This SGBM module explicitly follows from the Rectified Pair module and
  assumes inputs are rectified grayscale images (same size & epipolar-aligned).
  Downstream modules that will be integrated:
    - Postprocess: speckle filtering, WLS guided filter, LR-consistency
    - Depth: disparity-to-depth conversion (metric)
    - Vizualisation/3D: overlay, point-cloud generation, mesh or odometry
"""

def build_sgbm(
    min_disp=0, num_disp=128, block_size=5,
    disp12_max_diff=1, prefilter_cap=31, uniqueness_ratio=10,
    speckle_window_size=50, speckle_range=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
):
    channels = 1
    P1 = 8 * channels * (block_size ** 2)
    P2 = 32 * channels * (block_size ** 2)
    return cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=disp12_max_diff,
        preFilterCap=prefilter_cap,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        mode=mode,
    )

def compute_disparity(rectified_L, rectified_R, sgbm):
    # Inputs come from Rectify module
    disp_fixed = sgbm.compute(rectified_L, rectified_R)
    return disp_fixed.astype(np.float32) / 16.0

def disparity_to_depth(disp, f_pixels, baseline_m, eps=1e-6):
    Z = (f_pixels * baseline_m) / (disp + eps)
    Z[disp <= eps] = 0.0
    return Z

def run_sgbm_stage(rectified_L, rectified_R, f_pixels=None, baseline_m=None, sgbm=None):
    """
    Entrypoint for pipeline stage:
      Inputs: rectified_L, rectified_R (grayscale) from Rectify module
      Outputs:
        - disparity_px (float32): to Postprocess module 
        - depth_m (float32, optional): to Viz/3D module (point cloud) if f & B provided
    """
    if sgbm is None:
        sgbm = build_sgbm()
    disparity_px = compute_disparity(rectified_L, rectified_R, sgbm)

    depth_m = None
    if f_pixels is not None and baseline_m is not None:
        depth_m = disparity_to_depth(disparity_px, f_pixels, baseline_m)

    return disparity_px, depth_m
