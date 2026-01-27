"""Input utilities for loading stereo image pairs and live streams."""

from typing import Generator, Iterable, Tuple, Union

import cv2
import numpy as np


def load_stereo_pair(left_image_path, right_image_path, downscale_factor=1.0):
    """
    Load a stereo image pair from file paths.
    
    Parameters:
    -----------
    left_image_path : str
        Path to the left image
    right_image_path : str
        Path to the right image
    
    Returns:
    --------
    left_img_rgb : np.ndarray
        Left image in RGB format
    right_img_rgb : np.ndarray
        Right image in RGB format
    """
    # Load images
    left_img = cv2.imread(left_image_path)
    right_img = cv2.imread(right_image_path)

    if left_img is None or right_img is None:
        raise FileNotFoundError("One or both image paths are invalid.")

    # Convert from BGR (OpenCV default) to RGB for Matplotlib display
    left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    # Downscale if needed
    if downscale_factor != 1.0:
        new_size_left = (int(left_img_rgb.shape[1] * downscale_factor), int(left_img_rgb.shape[0] * downscale_factor))
        new_size_right = (int(right_img_rgb.shape[1] * downscale_factor), int(right_img_rgb.shape[0] * downscale_factor))
        left_img_rgb = cv2.resize(left_img_rgb, new_size_left, interpolation=cv2.INTER_AREA)
        right_img_rgb = cv2.resize(right_img_rgb, new_size_right, interpolation=cv2.INTER_AREA)

    return left_img_rgb, right_img_rgb


    return left_img_rgb, right_img_rgb


# --- Live video helpers ---

def open_capture(source: Union[int, str]) -> cv2.VideoCapture:
    """Open a cv2.VideoCapture from camera index, file path, or URL."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {source}")
    return cap


def _read_frame(cap: cv2.VideoCapture, downscale_factor: float) -> np.ndarray:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from video source")
    if downscale_factor != 1.0:
        new_size = (
            int(frame.shape[1] * downscale_factor),
            int(frame.shape[0] * downscale_factor),
        )
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
    return frame


def stereo_stream(
    left_source: Union[int, str],
    right_source: Union[int, str],
    downscale_factor: float = 1.0,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield synchronized frames from two captures.

    The generator raises StopIteration when either stream ends. Caller is
    responsible for releasing captures when finished.
    """
    if downscale_factor <= 0 or downscale_factor > 1.0:
        raise ValueError("downscale_factor must be between 0 and 1.")

    cap_L = open_capture(left_source)
    cap_R = open_capture(right_source)

    try:
        while True:
            left = _read_frame(cap_L, downscale_factor)
            right = _read_frame(cap_R, downscale_factor)
            yield left, right
    finally:
        cap_L.release()
        cap_R.release()
