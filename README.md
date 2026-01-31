# Depthlib
A python library for depth estimation using stereo vision

## Install
```bash
pip install depthlib
```

## Development Environment Setup

Follow these steps to prepare the project environment:

1. **Create a virtual environment**
    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment**
    - Windows
      ```bash
      venv\Scripts\activate
      ```
    - macOS/Linux
      ```bash
      source venv/bin/activate
      ```

3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    - CPU-only PyTorch
      ```bash
      pip install torch
      ```
    - CUDA PyTorch (pick the right CUDA wheel from PyTorch site)
      ```bash
      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
      ```
    See https://pytorch.org/get-started/locally/ to choose the correct CUDA version.

### Run Examples
- For stereo images - example_stereo.py
- For stereo video - example_stereo_live.py

> [!Note]
> Download demo left and right videos from [here](https://drive.google.com/drive/folders/1bNGE9a86ZHHI8yMr0GGY3mHU0LQ8qmYK?usp=sharing)
> and put it inside assets folder.