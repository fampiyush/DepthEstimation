import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
from pathlib import Path

# --- 1. DYNAMIC LIBRARY LOADING ---
# This block ensures the code works on any machine.
# It looks for 'libsgm_wrapper.so' in the '../lib' folder relative to this script.

HAS_GPU_LIB = False
try:
    # Get path to 'DepthEstimation/lib' relative to this file
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    lib_path = project_root / "lib"

    # Add to system path so Python can find the module
    if lib_path.exists() and str(lib_path) not in sys.path:
        sys.path.append(str(lib_path))
    
    # Fallback for Colab/Dev environments
    colab_path = Path("/content")
    if colab_path.exists() and str(colab_path) not in sys.path:
        sys.path.append(str(colab_path))

    # Attempt Import
    import libsgm_wrapper
    HAS_GPU_LIB = True
    print(f"✅ [SGM] GPU Library loaded successfully.")

except ImportError:
    # This allows the code to import without crashing, but compute() will fail if called.
    print(f"⚠️ [SGM] WARNING: 'libsgm_wrapper.so' not found in {lib_path}")
    print("   GPU acceleration will be unavailable.")

# --- 2. GPU ALGORITHM HELPERS ---

def sanitize_memory_for_gpu(img, align=128):
    """
    Prepares an image for the libSGM GPU kernel.
    
    CRITICAL SAFETY FUNCTION:
    1. Align Width: libSGM crashes if image width is not a multiple of 128/64.
       We pad the image to the nearest 128 pixels.
    2. Contiguous Memory: C++ pointers require a solid block of memory. 
       We use np.ascontiguousarray() to prevent SegFaults.
       
    Args:
        img: Input grayscale image (numpy array)
        align: Alignment block size (default 128)
        
    Returns:
        padded_img: Safe image for GPU
        (h, w): Original dimensions for cropping later
    """
    h, w = img.shape[:2]
    new_w = int(np.ceil(w / align) * align)
    new_h = int(np.ceil(h / align) * align)
    
    pad_h = new_h - h
    pad_w = new_w - w
    
    # Pad with zeros (black border)
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    # Force memory continuity
    return np.ascontiguousarray(padded), (h, w)

# --- 3. MAIN PIPELINE ---

def run_production_sgm(imgL, imgR, scale=0.35):
    """
    Executes the optimized GPU SGM pipeline.
    
    Strategy:
    1. Scale: 0.35x (Sweet spot for 100ms latency vs quality)
    2. Compute: Single-pass SGM on GPU
    3. Clean: Median + Morphological filters to remove noise
    
    Args:
        imgL, imgR: Rectified stereo pair (Grayscale or Color)
        scale: Resize factor (default 0.35 for speed)
        
    Returns:
        disp_final: Disparity map at ORIGINAL resolution (float32)
        time_ms: Total execution time in milliseconds
    """
    if not HAS_GPU_LIB:
        raise RuntimeError("GPU Library not loaded.")

    t0 = time.time()

    # A. PREPROCESS
    h_orig, w_orig = imgL.shape[:2]
    target_w, target_h = int(w_orig * scale), int(h_orig * scale)
    
    # Resize (Linear is good balance of speed/smoothness)
    l_small = cv2.resize(imgL, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    r_small = cv2.resize(imgR, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    
    # B. GPU PREPARATION (Padding)
    # We pad the *scaled* image to 128 alignment
    l_input, (h_pad, w_pad) = sanitize_memory_for_gpu(l_small)
    r_input, _ = sanitize_memory_for_gpu(r_small)
    
    # C. GPU EXECUTION
    # num_disparities=128: Sufficient for 0.35x scale (covers ~365px in full res)
    disp_raw = libsgm_wrapper.compute_disparity(l_input, r_input, num_disparities=128)
    
    # D. POST-PROCESS
    # 1. Crop padding
    disp = disp_raw[:h_pad, :w_pad]
    
    # 2. Mask Invalid Values
    # libSGM outputs 65535 (0xFFFF) for invalid pixels. We set them to 0.
    disp[disp > 60000] = 0
    
    # 3. Convert to float (Subpixel precision / 16.0)
    disp_f = disp.astype(np.float32) / 16.0
    
    # 4. Fast Median Filter (3x3)
    # Removes "salt and pepper" noise speckles efficiently
    disp_clean = cv2.medianBlur(disp_f, 3)
    
    # 5. Upscale to Original Resolution
    # We scale the image size AND the disparity values
    disp_final = cv2.resize(disp_clean, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    disp_final = disp_final * (1.0 / scale)
    
    t1 = time.time()
    return disp_final, (t1-t0)*1000

# --- 4. VISUALIZATION HELPER ---

def visualize_result(disp, time_ms, title="Final Result"):
    """
    Generates a professional 'Turbo' colormap visualization.
    """
    plt.figure(figsize=(14, 8))
    
    # Smart Contrast Stretching
    # We ignore the top 2% of outliers to make the colors pop
    valid_pixels = disp[disp > 0]
    vmax = np.percentile(valid_pixels, 98) if len(valid_pixels) > 0 else 100
    
    plt.imshow(disp, cmap='turbo_r', vmin=0, vmax=vmax)
    plt.colorbar(label="Disparity (Pixels)")
    plt.title(f"{title} (GPU + CPU: {time_ms:.1f}ms)")
    plt.axis('off')
    plt.show()

# --- 5. TEST RUNNER (Runs only if executed directly) ---
if __name__ == "__main__":
    # Look for test images in current folder or assets/
    # Adjust paths as needed for your local machine
    search_paths = ["rectified_left.png", "assets/rectified_left.png", "../assets/rectified_left.png"]
    left_path = next((p for p in search_paths if os.path.exists(p)), None)
    
    if left_path:
        right_path = left_path.replace("left", "right")
        if os.path.exists(right_path):
            print(f"🚀 Running Test on: {left_path}")
            
            imgL = cv2.imread(left_path, 0) # Load Grayscale
            imgR = cv2.imread(right_path, 0)
            
            try:
                # Run Pipeline
                disp, ms = run_production_sgm(imgL, imgR)
                print(f"✅ Success! Pipeline Latency: {ms:.2f} ms")
                
                # Visualize
                visualize_result(disp, ms)
            except Exception as e:
                print(f"❌ Pipeline Failed: {e}")
        else:
            print("❌ Right image not found.")
    else:
        print("ℹ️  No test images found. Import this module to use `run_production_sgm`.")