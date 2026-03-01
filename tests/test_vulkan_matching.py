#!/usr/bin/env python3
"""
Tests for vulkan_tensor_matching Python bindings.

Validates the CPU template matching implementation against OpenCV.
Vulkan GPU tests are included but will skip if Vulkan is unavailable.

The CPU implementation is the reference - Vulkan should match its results.
"""

import pytest
import numpy as np
from pathlib import Path

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("WARNING: OpenCV not available - comparison tests will fail")

from rust_python_lib import (
    ImageData,
    TemplateMatch,
    VulkanTensorMatcher,
    MatchTemplateMethod,
    match_template_cpu,
)


TEST_ASSETS_DIR = Path(__file__).parent.parent / "test_assets"


def numpy_to_image_data(arr: np.ndarray) -> ImageData:
    """Convert numpy array to ImageData."""
    arr = arr.astype(np.float32)
    return ImageData(
        data=arr.flatten().tolist(),
        width=arr.shape[1],
        height=arr.shape[0],
        channels=1,
    )


def image_data_to_numpy(image_data: ImageData) -> np.ndarray:
    """Convert ImageData to numpy array."""
    return np.array(image_data.data, dtype=np.float32).reshape(
        (image_data.height, image_data.width)
    )


class TestCPUvsOpenCV:
    """
    Validate CPU template matching against OpenCV.
    
    These tests ensure our Rust CPU implementation produces results
    consistent with OpenCV's well-tested template matching functions.
    """
    
    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV required")
    def test_sse_matches_opencv_sqdiff(self):
        """
        Sum of Squared Errors should match OpenCV TM_SQDIFF.
        
        Both find minimum at the best match location.
        """
        np.random.seed(42)
        image = np.random.rand(60, 60).astype(np.float32)
        template = image[20:30, 20:30].copy()  # Exact match exists
        
        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)
        
        # CPU SSE
        cpu_result = match_template_cpu(
            img_data, tmpl_data,
            MatchTemplateMethod.sum_of_squared_errors()
        )
        cpu_arr = image_data_to_numpy(cpu_result)
        cpu_min_y, cpu_min_x = np.unravel_index(np.argmin(cpu_arr), cpu_arr.shape)
        
        # OpenCV SQDIFF
        cv_result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)
        cv_min_y, cv_min_x = np.unravel_index(np.argmin(cv_result), cv_result.shape)
        
        # Both should find exact match at (20, 20)
        assert abs(cpu_min_x - 20) <= 1, f"CPU SSE wrong: ({cpu_min_x}, {cpu_min_y})"
        assert abs(cv_min_x - 20) <= 1, f"OpenCV wrong: ({cv_min_x}, {cv_min_y})"
    
    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV required")
    def test_cross_correlation_matches_opencv_ccorr(self):
        """
        Cross-correlation should match OpenCV TM_CCORR.
        
        Both find maximum at the best match location.
        """
        np.random.seed(456)
        image = np.random.rand(50, 50).astype(np.float32) * 0.5
        template = np.random.rand(10, 10).astype(np.float32) * 0.3 + 0.2
        
        # Plant template at known location (brighter)
        image[25:35, 25:35] = template * 1.5
        
        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)
        
        # CPU cross-correlation
        cpu_result = match_template_cpu(
            img_data, tmpl_data,
            MatchTemplateMethod.cross_correlation()
        )
        cpu_arr = image_data_to_numpy(cpu_result)
        cpu_max_y, cpu_max_x = np.unravel_index(np.argmax(cpu_arr), cpu_arr.shape)
        
        # OpenCV CCORR
        cv_result = cv2.matchTemplate(image, template, cv2.TM_CCORR)
        cv_max_y, cv_max_x = np.unravel_index(np.argmax(cv_result), cv_result.shape)
        
        # Both should find peak at (25, 25)
        assert abs(cpu_max_x - 25) <= 2, f"CPU wrong: ({cpu_max_x}, {cpu_max_y})"
        assert abs(cv_max_x - 25) <= 2, f"OpenCV wrong: ({cv_max_x}, {cv_max_y})"
    
    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV required")
    def test_output_shape_matches_opencv(self):
        """Output dimensions should match OpenCV for all methods."""
        image = np.random.rand(80, 80).astype(np.float32)
        template = np.random.rand(15, 15).astype(np.float32)
        
        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)
        
        expected_h = image.shape[0] - template.shape[0] + 1
        expected_w = image.shape[1] - template.shape[1] + 1
        
        # Test all methods
        methods = [
            (MatchTemplateMethod.sum_of_squared_errors(), "SSE"),
            (MatchTemplateMethod.cross_correlation(), "CCORR"),
        ]
        
        for method, name in methods:
            cpu_result = match_template_cpu(img_data, tmpl_data, method)
            cpu_arr = image_data_to_numpy(cpu_result)
            
            assert cpu_arr.shape == (expected_h, expected_w), \
                f"{name}: expected {(expected_h, expected_w)}, got {cpu_arr.shape}"
    
    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV required")
    @pytest.mark.skipif(not TEST_ASSETS_DIR.exists(), reason="No test assets")
    def test_lenna_template_matching(self):
        """Test on real Lenna image with template."""
        img_path = TEST_ASSETS_DIR / "lenna.png"
        tmpl_path = TEST_ASSETS_DIR / "templates" / "test1.png"
        
        if not img_path.exists() or not tmpl_path.exists():
            pytest.skip("Test images not found")
        
        # Load with OpenCV for fair comparison
        cv_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        cv_template = cv2.imread(str(tmpl_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        img_data = ImageData.from_file(str(img_path))
        tmpl_data = ImageData.from_file(str(tmpl_path))
        
        # CPU matching
        cpu_result = match_template_cpu(
            img_data, tmpl_data,
            MatchTemplateMethod.cross_correlation()
        )
        cpu_arr = image_data_to_numpy(cpu_result)
        
        # OpenCV matching
        cv_result = cv2.matchTemplate(cv_image, cv_template, cv2.TM_CCORR)
        
        # Both should produce valid output
        assert cpu_arr.shape == cv_result.shape, \
            f"Shape mismatch: CPU={cpu_arr.shape}, OpenCV={cv_result.shape}"
        assert np.all(np.isfinite(cpu_arr)), "CPU result contains NaN or Inf"
        assert np.all(np.isfinite(cv_result)), "OpenCV result contains NaN or Inf"


class TestVulkanGPU:
    """
    Vulkan GPU-accelerated template matching tests.
    
    These tests verify the Vulkan implementation when available.
    They will SKIP on systems without Vulkan support (expected on most macOS).
    """
    
    @pytest.fixture
    def matcher(self):
        """Create Vulkan matcher, skip if unavailable."""
        try:
            m = VulkanTensorMatcher()
            return m
        except ValueError as e:
            pytest.skip(f"Vulkan unavailable: {e}")
    
    @pytest.mark.skipif(not TEST_ASSETS_DIR.exists(), reason="No test assets")
    def test_vulkan_initialization(self, matcher):
        """Test that Vulkan matcher can be created."""
        assert matcher is not None
    
    @pytest.mark.skipif(not TEST_ASSETS_DIR.exists(), reason="No test assets")
    def test_vulkan_lenna_matching(self, matcher):
        """Test Vulkan matching on Lenna image."""
        img_path = TEST_ASSETS_DIR / "lenna.png"
        tmpl_path = TEST_ASSETS_DIR / "templates" / "test1.png"
        
        if not img_path.exists() or not tmpl_path.exists():
            pytest.skip("Test images not found")
        
        img_data = ImageData.from_file(str(img_path))
        tmpl_data = ImageData.from_file(str(tmpl_path))
        
        # Vulkan matching
        matches = matcher.match_template(
            img_data, tmpl_data,
            correlation_threshold=0.3,
            max_matches=10
        )
        
        # Validate results
        assert isinstance(matches, list)
        for match in matches:
            assert 0 <= match.x <= img_data.width
            assert 0 <= match.y <= img_data.height
            assert 0.0 <= match.correlation <= 1.0
    
    def test_vulkan_synthetic_pattern(self, matcher):
        """Test Vulkan finds planted pattern in synthetic image."""
        np.random.seed(789)
        image = np.random.rand(100, 100).astype(np.float32) * 0.3
        template = np.random.rand(12, 12).astype(np.float32) * 0.4 + 0.5
        image[40:52, 40:52] = template
        
        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)
        
        matches = matcher.match_template(
            img_data, tmpl_data,
            correlation_threshold=0.2,
            max_matches=5
        )
        
        # Should find at least one match
        assert len(matches) > 0, "Vulkan found no matches"
        
        # Best match should be near the planted location (40+6, 40+6) = (46, 46)
        best = matches[0]
        dx = abs(best.x - 46)
        dy = abs(best.y - 46)
        
        # Allow some tolerance for coordinate system differences
        assert dx + dy < 15, f"Match too far from expected: ({best.x}, {best.y})"


class TestImageData:
    """Basic ImageData functionality tests."""
    
    def test_load_from_file(self):
        """Test loading image from file."""
        img_path = TEST_ASSETS_DIR / "lenna.png"
        if not img_path.exists():
            pytest.skip("Lenna not found")
        
        img = ImageData.from_file(str(img_path))
        assert img.width > 0
        assert img.height > 0
        assert len(img.data) == img.width * img.height
    
    def test_creation_validation(self):
        """Test ImageData validates dimensions."""
        with pytest.raises(ValueError):
            ImageData(data=[0.0, 0.5], width=2, height=2, channels=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
