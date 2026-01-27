import pytest
from unittest.mock import patch
from depthlib.input import load_stereo_pair

def test_missing_file_error():
    """
    Ensure the system raises FileNotFoundError when images don't exist.
    """
    # We patch cv2.imread to return None (simulating a missing file)
    with patch('cv2.imread', return_value=None):
        with pytest.raises(FileNotFoundError) as excinfo:
            load_stereo_pair("fake_left.jpg", "fake_right.jpg")
        
    assert "One or both image paths are invalid" in str(excinfo.value)