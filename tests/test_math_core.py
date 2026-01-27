import numpy as np
import pytest
from depthlib.stereo_core import StereoCore

def test_depth_conversion_logic():
    """
    Mathematically verify: Z = (f * B) / d
    """
    core = StereoCore()
    
    # Setup known values
    f_px = 1000.0
    baseline_m = 0.5
    disparity_val = 100.0
    
    # Expected Depth: (1000 * 0.5) / 100 = 5.0 meters
    expected_depth = 5.0
    
    # Create a dummy 1x1 disparity map
    dummy_disp = np.array([[disparity_val]], dtype=np.float32)
    
    # Run conversion
    depth_map = core.disparity_to_depth(dummy_disp, f_px, baseline_m)
    
    # Assert
    assert depth_map.shape == (1, 1)
    assert np.isclose(depth_map[0,0], expected_depth), \
        f"Math failure: Expected {expected_depth}, got {depth_map[0,0]}"