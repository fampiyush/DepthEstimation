import numpy as np
import pytest
from depthlib.StereoDepthEstimator import StereoDepthEstimator
from unittest.mock import MagicMock

def test_full_pipeline_smoke_test():
    """
    Run the full estimate_depth() pipeline with mocked inputs to ensure no crashes.
    """
    # 1. Setup Estimator
    estimator = StereoDepthEstimator(downscale_factor=1.0)
    
    # 2. Inject fake rectified images directly into the core
    #    (Bypassing load_stereo_pair to avoid needing files)
    fake_img = np.zeros((480, 640), dtype=np.uint8)
    estimator.core.left_rectified = fake_img
    estimator.core.right_rectified = fake_img
    
    # 3. Configure minimal params to avoid errors
    estimator.configure_sgbm(
        min_disp=0, num_disp=16, block_size=3,
        focal_length=1000, baseline=0.5
    )
    
    # 4. Manually trigger processing (Mocking the load step)
    #    We call process_pair directly or mock the sources
    disparity, depth = estimator.core._process_pair(fake_img, fake_img)
    
    # 5. Assert results exist
    assert disparity is not None
    assert depth is not None
    assert disparity.shape == (480, 640)
    assert depth.shape == (480, 640)