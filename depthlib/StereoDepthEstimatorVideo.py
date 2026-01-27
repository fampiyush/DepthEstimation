from depthlib.input import stereo_stream, open_capture
from depthlib.visualizations import visualize_depth_live_gray, visualize_depth_live
from depthlib.stereo_core import StereoCore
from depthlib.threaded_stereo import ThreadedStereoCapture
import cv2
import time


class StereoDepthEstimatorVideo:
    '''class for estimating depth from stereo video streams'''

    def __init__(
        self,
        left_source=None, # Path to left video
        right_source=None, # Path to right video
        downscale_factor=1.0,
        visualize_live=False,
        saving_path=None, # Path to save output video
        fast_mode=False,  # Enable fast mode for higher FPS
        use_threading=True,  # Use threaded frame capture
        target_fps=30,  # Maximum FPS to process (0 = unlimited)
        drop_frames=False,  # Drop frames when processing is slow (True for live cameras, False for video files)
        visualize_gray=False,
    ) -> None:
        '''Initialize the StereoDepthEstimatorVideo with video sources and parameters.'''
        self.left_source = left_source
        self.right_source = right_source
        self.downscale_factor = downscale_factor
        self.visualize_live = visualize_live
        self.saving_path = saving_path
        self.fast_mode = fast_mode
        self.use_threading = use_threading
        self.target_fps = target_fps
        self._frame_interval = 1.0 / target_fps if target_fps > 0 else 0
        self.drop_frames = drop_frames
        self.visualize_gray = visualize_gray
        self.core = StereoCore(downscale_factor=downscale_factor, fast_mode=fast_mode)

    def configure_sgbm(self, **kwargs):
        """
        Configure SGBM parameters and rebuild matcher.
        
        Parameters:
        -----------
        min_disp : int, optional
            Minimum disparity (default: 0)
        num_disp : int, optional
            Number of disparities - must be divisible by 16 (default: 128)
        block_size : int, optional
            Block size for matching (default: 5)
        disp12_max_diff : int, optional
            Maximum allowed difference in left-right disparity check (default: 1)
        prefilter_cap : int, optional
            Prefilter cap (default: 31)
        uniqueness_ratio : int, optional
            Uniqueness ratio (default: 10)
        speckle_window_size : int, optional
            Speckle window size (default: 50)
        speckle_range : int, optional
            Speckle range (default: 2)
        
        Example:
        --------
        >>> estimator.configure_sgbm(num_disp=144, block_size=7)
        >>> estimator.configure_sgbm(min_disp=16, uniqueness_ratio=15)
        """
        self.core.configure_sgbm(**kwargs)

    def estimate_depth(self):
        '''Estimate depth from the stereo video streams.

        Yields:
            numpy.ndarray: depth map (meters) for each frame.
        '''
        if self.left_source is None or self.right_source is None:
            raise ValueError("Both left_source and right_source must be provided for video depth estimation.")

        self.core.configure_sgbm(**self.core.get_sgbm_params())

        # Allow window resizing
        cv2.namedWindow("Depth (live)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth (live)", 960, 540)
        
        if self.use_threading:
            # Use threaded capture for better FPS
            capture = ThreadedStereoCapture(
                self.left_source, self.right_source, 
                downscale_factor=self.downscale_factor,
                drop_frames=self.drop_frames
            )
            capture.start()
            
            try:
                frame_start_time = time.time()
                while True:
                    frame_pair = capture.read()
                    if frame_pair is None:
                        break
                    
                    left_frame, right_frame = frame_pair
                    disparity_px, depth_m = self.core.estimate_depth(left_frame, right_frame)

                    yield depth_m

                    if self.visualize_live:
                        if self.visualize_gray:
                            visualize_depth_live_gray(depth_m, self.target_fps)
                        else:
                            visualize_depth_live(depth_m, self.target_fps)

                    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
                        break
                    
                    # Enforce target FPS by sleeping if we're too fast
                    if self._frame_interval > 0:
                        elapsed = time.time() - frame_start_time
                        sleep_time = self._frame_interval - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    frame_start_time = time.time()
            finally:
                capture.stop()
                cv2.destroyAllWindows()
        else:
            # Original non-threaded path
            frame_start_time = time.time()
            for left_frame, right_frame in stereo_stream(self.left_source, self.right_source, downscale_factor=self.downscale_factor):
                disparity_px, depth_m = self.core.estimate_depth(left_frame, right_frame)

                yield depth_m

                if self.visualize_live:
                    if self.visualize_gray:
                        visualize_depth_live_gray(depth_m, self.target_fps)
                    else:
                        visualize_depth_live(depth_m, self.target_fps)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
                    cv2.destroyAllWindows()
                    break
                
                # Enforce target FPS by sleeping if we're too fast
                if self._frame_interval > 0:
                    elapsed = time.time() - frame_start_time
                    sleep_time = self._frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                frame_start_time = time.time()