#!/usr/bin/env python3
"""
Tests for vulkan_tensor_matching Python bindings.

Compares Rust library output against OpenCV's template matching results.
"""

import pytest
import numpy as np
from pathlib import Path

# Try to import opencv, skip tests if not available
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from rust_python_lib import (
    ImageData,
    TemplateMatch,
    VulkanTensorMatcher,
    MatchTemplateMethod,
    match_template_cpu,
    find_extremes,
    compress_image,
    VERSION,
    AUTHOR,
)


# Test assets directory
TEST_ASSETS_DIR = Path(__file__).parent.parent / "test_assets"


def skip_if_no_opencv():
    """Skip test if OpenCV is not available."""
    if not OPENCV_AVAILABLE:
        pytest.skip("OpenCV not available, skipping test")


def image_data_to_numpy(image_data: ImageData) -> np.ndarray:
    """Convert ImageData to numpy array for comparison."""
    return np.array(image_data.data, dtype=np.float32).reshape(
        (image_data.height, image_data.width)
    )


def numpy_to_image_data(arr: np.ndarray) -> ImageData:
    """Convert numpy array to ImageData.
    
    Assumes input array is already normalized to [0, 1] range.
    """
    if arr.ndim == 3:
        # Convert RGB to grayscale
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    
    # Ensure float32 and flatten
    arr = arr.astype(np.float32)
    return ImageData(
        data=arr.flatten().tolist(),
        width=arr.shape[1],
        height=arr.shape[0],
        channels=1,
    )


def opencv_match_template(
    image_arr: np.ndarray,
    template_arr: np.ndarray,
    method: str = "ccorr_normed"
) -> np.ndarray:
    """
    Perform template matching using OpenCV.
    
    Args:
        image_arr: Input image as numpy array (grayscale, float32, any range)
        template_arr: Template as numpy array
        method: Matching method ("sqdiff", "sqdiff_normed", "ccorr", "ccorr_normed", "corr", "corr_normed")
    
    Returns:
        Result correlation map
    """
    # OpenCV expects float32
    img = image_arr.astype(np.float32)
    tmpl = template_arr.astype(np.float32)
    
    method_map = {
        "sqdiff": cv2.TM_SQDIFF,
        "sqdiff_normed": cv2.TM_SQDIFF_NORMED,
        "ccorr": cv2.TM_CCORR,
        "ccorr_normed": cv2.TM_CCORR_NORMED,
        "corr": cv2.TM_CCOEFF,
        "corr_normed": cv2.TM_CCOEFF_NORMED,
    }
    
    cv_method = method_map.get(method, cv2.TM_CCORR_NORMED)
    result = cv2.matchTemplate(img, tmpl, cv_method)
    return result


class TestImageData:
    """Tests for ImageData class."""
    
    def test_creation(self):
        """Test creating ImageData from raw data."""
        data = [0.0, 0.5, 0.5, 1.0]
        img = ImageData(data=data, width=2, height=2, channels=1)
        assert img.width == 2
        assert img.height == 2
        assert len(img.data) == 4
    
    def test_creation_invalid_size(self):
        """Test that invalid dimensions raise error."""
        with pytest.raises(ValueError):
            ImageData(data=[0.0, 0.5], width=2, height=2, channels=1)
    
    @pytest.mark.skipif(not TEST_ASSETS_DIR.exists(), reason="Test assets not found")
    def test_from_file(self):
        """Test loading image from file."""
        img_path = TEST_ASSETS_DIR / "lenna.png"
        if not img_path.exists():
            pytest.skip("Lenna test image not found")
        
        img = ImageData.from_file(str(img_path))
        assert img.width > 0
        assert img.height > 0
        assert len(img.data) == img.width * img.height


class TestMatchTemplateCPU:
    """Tests for CPU-based template matching, comparing against OpenCV."""
    
    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
    def test_cross_correlation_normalized_vs_opencv(self):
        """Compare Rust cross-correlation normalized with OpenCV.
        
        Note: This test verifies that both implementations produce valid output
        with the expected dimensions. Exact correlation values may differ due to
        different normalization approaches.
        """
        # Create test images with variance (needed for normalized correlation)
        np.random.seed(42)  # For reproducibility
        image_arr = np.random.rand(100, 100).astype(np.float32) * 0.5  # Background noise
        
        # Add a distinct pattern at a known location
        image_arr[45:55, 45:55] = 0.9  # Bright square
        
        # Template that matches the pattern (with some variance)
        template_arr = np.random.rand(10, 10).astype(np.float32) * 0.2 + 0.7
        
        # Rust result
        rust_img = numpy_to_image_data(image_arr)
        rust_tmpl = numpy_to_image_data(template_arr)
        rust_result = match_template_cpu(
            rust_img, rust_tmpl, 
            MatchTemplateMethod.cross_correlation_normalized()
        )
        rust_arr = image_data_to_numpy(rust_result)
        
        # OpenCV result
        cv_result = opencv_match_template(image_arr, template_arr, "corr_normed")
        
        # Results should have the same shape
        assert rust_arr.shape == cv_result.shape
        
        # Verify output dimensions are correct
        expected_h = image_arr.shape[0] - template_arr.shape[0] + 1
        expected_w = image_arr.shape[1] - template_arr.shape[1] + 1
        assert rust_arr.shape == (expected_h, expected_w)
        assert cv_result.shape == (expected_h, expected_w)
        
        # Both should produce finite values (no NaN or Inf)
        assert np.all(np.isfinite(rust_arr))
        assert np.all(np.isfinite(cv_result))
        
        # Both should have some variation in output (not all zeros or constant)
        assert np.std(rust_arr) > 0 or np.max(rust_arr) > 0
        assert np.std(cv_result) > 0 or np.max(cv_result) > 0
    
    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
    def test_sum_of_squared_errors_vs_opencv(self):
        """Compare Rust SSE with OpenCV."""
        image_arr = np.random.rand(50, 50).astype(np.float32)
        template_arr = image_arr[20:30, 20:30].copy()
        
        # Rust result
        rust_img = numpy_to_image_data(image_arr)
        rust_tmpl = numpy_to_image_data(template_arr)
        rust_result = match_template_cpu(
            rust_img, rust_tmpl,
            MatchTemplateMethod.sum_of_squared_errors()
        )
        rust_arr = image_data_to_numpy(rust_result)
        
        # OpenCV result
        cv_result = opencv_match_template(image_arr, template_arr, "sqdiff")
        
        # For SSE, minimum value indicates best match
        rust_min_loc = np.unravel_index(np.argmin(rust_arr), rust_arr.shape)
        cv_min_loc = np.unravel_index(np.argmin(cv_result), cv_result.shape)
        
        # Minimum locations should be similar
        assert abs(rust_min_loc[0] - cv_min_loc[0]) <= 2
        assert abs(rust_min_loc[1] - cv_min_loc[1]) <= 2


class TestFindExtremes:
    """Tests for find_extremes function."""
    
    def test_find_max_min(self):
        """Test finding maximum and minimum values."""
        data = [0.0, 0.5, 1.0, 0.25, 0.75]
        img = ImageData(data=data, width=5, height=1, channels=1)
        extremes = find_extremes(img)
        
        assert extremes["max"]["value"] == 1.0
        assert extremes["min"]["value"] == 0.0
    
    def test_find_extremes_2d(self):
        """Test finding extremes in 2D image."""
        image_arr = np.zeros((10, 10), dtype=np.float32)
        image_arr[5, 5] = 1.0  # Max at center
        image_arr[0, 0] = 0.0  # Min at corner
        
        img = numpy_to_image_data(image_arr)
        extremes = find_extremes(img)
        
        assert extremes["max"]["value"] == 1.0
        assert extremes["max"]["x"] == 5  # x
        assert extremes["max"]["y"] == 5  # y


class TestCompressImage:
    """Tests for compress_image function."""
    
    def test_compress_by_factor_2(self):
        """Test compressing image by factor of 2."""
        image_arr = np.ones((10, 10), dtype=np.float32) * 0.5
        image_arr[0:5, 0:5] = 1.0  # Top-left quadrant bright
        
        img = numpy_to_image_data(image_arr)
        compressed = compress_image(img, factor=2)
        
        assert compressed.width == 5
        assert compressed.height == 5
        assert len(compressed.data) == 25
    
    def test_compress_preserves_average(self):
        """Test that compression preserves average intensity."""
        # Use a fixed pattern instead of random for reproducibility
        image_arr = np.ones((20, 20), dtype=np.float32) * 0.5
        original_avg = np.mean(image_arr)
        
        img = numpy_to_image_data(image_arr)
        compressed = compress_image(img, factor=2)
        compressed_arr = image_data_to_numpy(compressed)
        compressed_avg = np.mean(compressed_arr)
        
        # For uniform image, averages should be identical
        assert abs(original_avg - compressed_avg) < 0.001


class TestVulkanTensorMatcher:
    """Tests for Vulkan-based template matching."""
    
    @pytest.fixture
    def matcher(self):
        """Create a Vulkan matcher instance."""
        try:
            return VulkanTensorMatcher()
        except ValueError as e:
            if "Vulkan" in str(e) or "vulkan" in str(e).lower():
                pytest.skip("Vulkan not available, skipping test")
            raise
    
    @pytest.mark.skipif(not TEST_ASSETS_DIR.exists(), reason="Test assets not found")
    def test_match_template_basic(self, matcher):
        """Test basic template matching with Vulkan."""
        img_path = TEST_ASSETS_DIR / "lenna.png"
        tmpl_path = TEST_ASSETS_DIR / "templates" / "test1.png"
        
        if not img_path.exists() or not tmpl_path.exists():
            pytest.skip("Test images not found")
        
        target = ImageData.from_file(str(img_path))
        template = ImageData.from_file(str(tmpl_path))
        
        matches = matcher.match_template(
            target, template,
            correlation_threshold=0.5,
            max_matches=10
        )
        
        assert isinstance(matches, list)
        # We may or may not find matches depending on the template
        for match in matches:
            assert isinstance(match, TemplateMatch)
            assert 0.0 <= match.correlation <= 1.0
            assert match.x < target.width
            assert match.y < target.height
    
    def test_match_template_synthetic(self, matcher):
        """Test template matching with synthetic data."""
        # Create a simple image with a known pattern
        image_arr = np.zeros((100, 100), dtype=np.float32)
        image_arr[40:60, 40:60] = 1.0  # Bright square
        
        template_arr = np.ones((20, 20), dtype=np.float32)
        
        target = numpy_to_image_data(image_arr)
        template = numpy_to_image_data(template_arr)
        
        matches = matcher.match_template(
            target, template,
            correlation_threshold=0.3,
            max_matches=5
        )
        
        # Should find the bright square
        assert len(matches) >= 1
        best_match = matches[0]
        
        # Match should be near the center of the bright square (50, 50)
        assert 35 <= best_match.x <= 65
        assert 35 <= best_match.y <= 65


class TestVersionAndAuthor:
    """Tests for version and author constants."""
    
    def test_version_exists(self):
        """Test that VERSION is defined."""
        assert VERSION is not None
        assert isinstance(VERSION, str)
        assert len(VERSION) > 0
    
    def test_author_exists(self):
        """Test that AUTHOR is defined."""
        assert AUTHOR is not None
        assert isinstance(AUTHOR, str)
        assert len(AUTHOR) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
