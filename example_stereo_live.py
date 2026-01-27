"""Live stereo depth demo: capture -> SGBM -> display.

Usage examples:
- Two USB cams: set left_src = 0, right_src = 1
- Two video files: set paths to files
- RTSP/URL sources: set RTSP URLs
Press ESC to exit.
"""
from typing import Union

from depthlib import StereoDepthEstimatorVideo

def main():
    # Configure your sources here
    left_src: Union[int, str] = './assets/left.mp4'
    right_src: Union[int, str] = './assets/right.mp4'

    downscale = 0.7  # smaller -> faster

    ndisp = 128
    focal_length = 679.01
    baseline_mm = 572.5
    doffs = 0

    estimator = StereoDepthEstimatorVideo(
        left_source=left_src,
        right_source=right_src,
        downscale_factor=downscale,
        visualize_live=True,
        target_fps=30
    )
    estimator.configure_sgbm(
        num_disp=ndisp,
        focal_length=focal_length,
        baseline=baseline_mm / 1000.0,
        doffs=doffs,
        hole_filling=True,
    )

    # Consume the generator to get each depth map
    for depth_m in estimator.estimate_depth():
        # Use depth_m here (e.g., save, analyze, etc.)
        pass


if __name__ == "__main__":
    main()
