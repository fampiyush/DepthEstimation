import time
import numpy as np

if __name__ == "__main__":
    left_image_path = "./assets/unrectified_stereo/unrec_0.png"
    right_image_path = "./assets/unrectified_stereo/unrec_1.png"
    calib = parse_middlebury_calib("./assets/calib_unrectified.txt")  

    ndisp = 290 #256          # before downscale; will be scaled by 0.5 → 128
    focal_length = 3997.684
    baseline_mm = 111.53 #193.001
    doffs = 0 #131.111
    #cam0=[1758.23 0 953.34; 0 1758.23 552.29; 0 0 1]
    #cam1=[1758.23 0 953.34; 0 1758.23 552.29; 0 0 1]
    vmin=75
    vmax=262

    # ndisp = 128
    # focal_length = 679.01
    # baseline_mm = 572.5
    # doffs = 0

    # Using StereoDepthEstimator
    estimator = depthlib.StereoDepthEstimator(left_source=left_image_path, right_source=right_image_path, 
                                              downscale_factor=0.5)
    estimator.configure_sgbm(
        num_disp=ndisp,
        focal_length=focal_length,
        baseline=baseline_mm / 1000.0,
        doffs=doffs,
    )

    estimator.configure_sgbm(
        num_disp=ndisp,                    # or your adjusted value, e.g. 290 before downscale
        focal_length=calib['cam0'][0, 0],
        baseline=calib['baseline_m'],
        doffs=calib['doffs'],
        cam_matrix_L=calib['cam0'],            # full 3x3 intrinsic matrix for left
        cam_matrix_R=calib['cam1'],            # full 3x3 for right
        image_width=calib['width'],            # exact calibrated resolution
        image_height=calib['height'],
    )
    start_time = time.time()
    disparity_px, depth_m = estimator.estimate_depth()
    latency_ms = (time.time() - start_time) * 1000.0
    print(f"Depth estimation completed in {latency_ms:.2f} ms")

    estimator.visualize_results()

    invalid_value = -1.0
    valid_disp = disparity_px > invalid_value

    print("\n=== Raw Disparity Statistics ===")
    print(
        f"Disparity range: {disparity_px[valid_disp].min():.2f} - "
        f"{disparity_px[valid_disp].max():.2f} pixels"
    )
    print(f"Invalid disparities: {(~valid_disp).sum() / valid_disp.size * 100:.2f}%")
