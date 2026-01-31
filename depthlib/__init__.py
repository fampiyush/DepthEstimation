from depthlib.StereoDepthEstimator import StereoDepthEstimator
from depthlib.StereoDepthEstimatorVideo import StereoDepthEstimatorVideo
from depthlib.visualizations import (visualize_stereo_pair, visualize_disparity, 
                            visualize_depth, visualize_disparity_and_depth)

__all__ = [
    'StereoDepthEstimator',
    'visualize_stereo_pair',
    'visualize_disparity',
    'visualize_depth',
    'visualize_disparity_and_depth',
    'StereoDepthEstimatorVideo',
    'MonocularDepthEstimator',
]

def __getattr__(name):
    """Lazy import for MonocularDepthEstimator to avoid loading transformers/torch unnecessarily."""
    if name == "MonocularDepthEstimator":
        from depthlib.MonocularDepthEstimator import MonocularDepthEstimator
        return MonocularDepthEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")