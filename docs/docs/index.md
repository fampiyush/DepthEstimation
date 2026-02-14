# Depthlib

**Depthlib** is a Python library for depth estimation using stereo vision and monocular deep learning models.

The library provides simple, high-level APIs that abstract away internal image processing and stereo matching details, making it easy to estimate depth from images and video streams.

---

## Features

- **Stereo Depth Estimation** - Compute depth maps from rectified stereo image pairs using Semi-Global Block Matching (SGBM)
- **Live Video Depth** - Real-time depth estimation from synchronized stereo video streams or cameras
- **Monocular Depth** - Single-image depth estimation using pre-trained deep learning models (Depth Anything V2)
- **Configurable Pipeline** - Fine-tune SGBM parameters for optimal results in your scene
- **Visualization Tools** - Built-in functions for displaying disparity and depth maps

---

## Supported Modes

| Mode | Class | Description |
|------|-------|-------------|
| Stereo Images | `StereoDepthEstimator` | Depth from a single pair of stereo images |
| Stereo Video | `StereoDepthEstimatorVideo` | Real-time depth from video streams/cameras |
| Monocular | `MonocularDepthEstimator` | Depth from a single RGB image using AI |

---

## Quick Start

## Installation

```bash
pip install depthlib
```

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/depthlib.git

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import depthlib

# Stereo depth estimation
estimator = depthlib.StereoDepthEstimator(
    left_source='left.png',
    right_source='right.png',
    downscale_factor=0.5
)
estimator.configure_sgbm(
    num_disp=128,
    focal_length=3997.684,
    baseline=0.193
)
disparity, depth = estimator.estimate_depth()
estimator.visualize_results()
```

---

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- PyTorch (for monocular depth)
- Transformers (for monocular depth)

---

## Next Steps

- [Getting Started](getting_started.md) - Setup guide and first examples
- [Stereo Depth (Images)](stereo_image.md) - Full API for image-based stereo depth
- [Stereo Depth (Video)](stereo_video.md) - Real-time video depth estimation
- [SGBM Configuration](sgbm_config.md) - Tune stereo matching parameters
