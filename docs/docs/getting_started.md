# Getting Started

This guide will help you set up Depthlib and run your first depth estimation.

---

## Installation

### Quick Install (Recommended)

```bash
pip install depthlib
```

For GPU acceleration, also install PyTorch with CUDA support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

!!! tip "Finding Your CUDA Version"
    Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to select the correct CUDA version for your system.

---

### Development Installation

If you want to contribute or modify the library, follow these steps:

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/depthlib.git
cd depthlib
```

#### 2. Create a Virtual Environment

```bash
python -m venv venv
```

#### 3. Activate the Virtual Environment

=== "Windows"

    ```bash
    venv\Scripts\activate
    ```

=== "macOS/Linux"

    ```bash
    source venv/bin/activate
    ```

#### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 5. Install in Editable Mode

```bash
pip install -e .
```

#### 6. Install PyTorch

Choose the appropriate version based on your hardware:

**CPU-only:**
```bash
pip install torch
```

**CUDA (GPU):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

---

## Project Structure

```
depthlib/
├── __init__.py                    # Main exports
├── StereoDepthEstimator.py        # Stereo image depth
├── StereoDepthEstimatorVideo.py   # Stereo video depth
├── MonocularDepthEstimator.py     # AI-based monocular depth
├── stereo_core.py                 # SGBM implementation
├── visualizations.py              # Display utilities
├── input.py                       # Image/video loading
├── rectify.py                     # Stereo rectification
└── postprocess.py                 # Disparity filtering
```

---

## Your First Depth Estimation

### Stereo Images

```python
import depthlib

# Initialize estimator with stereo pair
estimator = depthlib.StereoDepthEstimator(
    left_source='./assets/stereo_pairs/im0.png',
    right_source='./assets/stereo_pairs/im1.png',
    downscale_factor=0.5
)

# Configure SGBM with camera parameters
estimator.configure_sgbm(
    num_disp=280,
    focal_length=3997.684,      # pixels
    baseline=0.193,             # meters
    doffs=131.111               # disparity offset
)

# Estimate depth
disparity_px, depth_m = estimator.estimate_depth()

# Visualize results
estimator.visualize_results()
```

### Live Video Stream

```python
from depthlib import StereoDepthEstimatorVideo

# Initialize with video sources (files or camera indices)
estimator = StereoDepthEstimatorVideo(
    left_source='./assets/left.mp4',
    right_source='./assets/right.mp4',
    downscale_factor=0.7,
    visualize_live=True,
    target_fps=30
)

# Configure SGBM
estimator.configure_sgbm(
    num_disp=128,
    focal_length=679.01,
    baseline=0.5725
)

# Process video frames
for depth_m in estimator.estimate_depth():
    # depth_m contains the depth map for each frame
    pass  # Press ESC to exit
```

### Monocular Depth (Single Image)

```python
import depthlib

# Initialize with pre-trained model
model_path = "models/hub/models--depth-anything--Depth-Anything-V2-Base-hf/snapshots/..."
estimator = depthlib.MonocularDepthEstimator(
    model_path=model_path,
    device='cuda',  # or 'cpu'
    downscale_factor=0.5
)

# Estimate depth from single image
depth_map = estimator.estimate_depth(image_path='./assets/image.png')

# Visualize
estimator.visualize_depth()
```

---

## Understanding Camera Parameters

For accurate metric depth from stereo vision, you need calibration parameters:

| Parameter | Description | Units |
|-----------|-------------|-------|
| `focal_length` | Camera focal length | pixels |
| `baseline` | Distance between camera centers | meters |
| `doffs` | Disparity offset (cx_right - cx_left) | pixels |
| `num_disp` | Maximum disparity range to search | pixels |

!!! note "Depth Formula"
    Depth is calculated as: $Z = \frac{f \cdot B}{d + d_{offs}}$
    
    Where $f$ is focal length, $B$ is baseline, and $d$ is disparity.

---

## Next Steps

- [StereoDepthEstimator API](stereo_image.md) - Complete reference for image-based depth
- [StereoDepthEstimatorVideo API](stereo_video.md) - Video/live camera depth
- [SGBM Configuration](sgbm_config.md) - Fine-tune matching parameters
- [Examples](examples.md) - More code examples
