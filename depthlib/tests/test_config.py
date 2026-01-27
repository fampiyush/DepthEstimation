import pytest
from depthlib.StereoDepthEstimator import StereoDepthEstimator

def test_sgbm_configuration_update():
    """
    Verify that user configurations successfully update the internal SGBM params.
    """
    estimator = StereoDepthEstimator(downscale_factor=0.5)
    
    # Default is typically 128 disparities
    
    # Change settings
    new_params = {
        'min_disp': 16,
        'num_disp': 64, # 64 * 0.5 = 32 internal
        'block_size': 7
    }
    estimator.configure_sgbm(**new_params)
    
    # Check internal state
    current_params = estimator.get_sgbm_params()
    
    assert current_params['min_disp'] == 16
    assert current_params['block_size'] == 7
    # Note: num_disp is scaled by downscale_factor internally
    assert current_params['num_disp'] == 32