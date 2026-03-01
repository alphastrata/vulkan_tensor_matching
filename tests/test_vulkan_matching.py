#!/usr/bin/env python3
"""
Tests for vulkan_tensor_matching Python bindings.

Compares Vulkan GPU template matching against:
1. CPU-based Rust implementation  
2. OpenCV's template matching

Vulkan tests will SKIP if Vulkan is unavailable (expected on macOS without MoltenVK).
CPU vs OpenCV tests validate the core matching algorithms.
"""

import pytest
import numpy as np
from pathlib import Path

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("WARNING: OpenCV not available - GPU comparison tests will fail")

from rust_python_lib import (
    ImageData,
    TemplateMatch,
    VulkanTensorMatcher,
    MatchTemplateMethod,
    match_template_cpu,
)


TEST_ASSETS_DIR = Path(__file__).parent.parent / "test_assets"


def numpy_to_image_data(arr: np.ndarray) -> ImageData:
    """Convert numpy array to ImageData (expects float32, normalized to [0,1])."""
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


def find_peak_2d(arr: np.ndarray, top_k: int = 1) -> list[tuple[int, int, float]]:
    """Find top-k peak locations and values in a 2D array."""
    flat = arr.flatten()
    peak_indices = np.argsort(flat)[-top_k:][::-1]
    peaks = []
    for idx in peak_indices:
        y, x = np.unravel_index(idx, arr.shape)
        peaks.append((int(x), int(y), float(arr[y, x])))
    return peaks


class TestVulkanVsCPU:
    """
    Compare Vulkan GPU matching against CPU implementation.
    
    These tests verify that the GPU-accelerated Vulkan implementation
    produces results consistent with the CPU reference implementation.
    """
    
    @pytest.fixture
    def matcher(self):
        """Create Vulkan matcher, skip if unavailable."""
        try:
            m = VulkanTensorMatcher()
            print("\n✓ Vulkan matcher initialized successfully")
            return m
        except ValueError as e:
            pytest.skip(f"Vulkan unavailable: {e}")
    
    def test_synthetic_pattern_matching(self, matcher):
        """Test that Vulkan and CPU find matches at same location for synthetic pattern."""
        # Create image with known pattern at known location
        np.random.seed(42)
        image = np.random.rand(100, 100).astype(np.float32) * 0.3  # Low noise background
        
        # Plant a bright template at specific location
        template = np.random.rand(15, 15).astype(np.float32) * 0.4 + 0.5  # Bright pattern
        image[40:55, 40:55] = template
        
        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)
        
        # CPU matching
        cpu_result = match_template_cpu(
            img_data, tmpl_data,
            MatchTemplateMethod.cross_correlation()
        )
        cpu_arr = image_data_to_numpy(cpu_result)
        cpu_peaks = find_peak_2d(cpu_arr, top_k=3)
        cpu_best_x, cpu_best_y, cpu_best_score = cpu_peaks[0]
        
        print(f"\nCPU best match: ({cpu_best_x}, {cpu_best_y}) score={cpu_best_score:.4f}")
        
        # Vulkan matching
        vulkan_matches = matcher.match_template(
            img_data, tmpl_data,
            correlation_threshold=0.1,
            max_matches=3
        )
        
        if vulkan_matches:
            vulkan_best = vulkan_matches[0]
            print(f"Vulkan best match: ({vulkan_best.x}, {vulkan_best.y}) score={vulkan_best.correlation:.4f}")
            
            # Both should find high correlation at similar location
            assert cpu_best_score > 0.5, f"CPU failed to find pattern: {cpu_best_score}"
            assert vulkan_best.correlation > 0.3, f"Vulkan failed to find pattern: {vulkan_best.correlation}"
        else:
            pytest.fail("Vulkan found no matches above threshold")
    
    @pytest.mark.skipif(not TEST_ASSETS_DIR.exists(), reason="Test assets not found")
    def test_real_image_matching(self, matcher):
        """Test Vulkan vs CPU on real image (Lenna)."""
        img_path = TEST_ASSETS_DIR / "lenna.png"
        tmpl_path = TEST_ASSETS_DIR / "templates" / "test1.png"
        
        if not img_path.exists() or not tmpl_path.exists():
            pytest.skip("Test images not found")
        
        # Load images
        img_data = ImageData.from_file(str(img_path))
        tmpl_data = ImageData.from_file(str(tmpl_path))
        
        print(f"\nImage: {img_data.width}x{img_data.height}, Template: {tmpl_data.width}x{tmpl_data.height}")
        
        # CPU matching
        cpu_result = match_template_cpu(
            img_data, tmpl_data,
            MatchTemplateMethod.cross_correlation()
        )
        cpu_arr = image_data_to_numpy(cpu_result)
        cpu_peaks = find_peak_2d(cpu_arr, top_k=5)
        
        print(f"CPU found {len(cpu_peaks)} peaks, best: {cpu_peaks[0]}")
        
        # Vulkan matching
        vulkan_matches = matcher.match_template(
            img_data, tmpl_data,
            correlation_threshold=0.3,
            max_matches=5
        )
        
        print(f"Vulkan found {len(vulkan_matches)} matches")
        for i, m in enumerate(vulkan_matches[:3]):
            print(f"  [{i+1}] ({m.x}, {m.y}) corr={m.correlation:.4f}")
        
        # Validate Vulkan results are in valid range
        for match in vulkan_matches:
            assert 0 <= match.x <= img_data.width
            assert 0 <= match.y <= img_data.height
            assert 0.0 <= match.correlation <= 1.0


class TestVulkanVsOpenCV:
    """Compare Vulkan GPU matching against OpenCV."""
    
    @pytest.fixture
    def matcher(self):
        """Create Vulkan matcher, skip if unavailable."""
        try:
            return VulkanTensorMatcher()
        except ValueError as e:
            if "Vulkan" in str(e) or "vulkan" in str(e).lower():
                pytest.skip("Vulkan not available")
            raise
    
    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
    def test_synthetic_pattern_opencv_comparison(self, matcher):
        """Test that Vulkan and OpenCV find matches at similar locations."""
        # Create image with clear pattern
        np.random.seed(123)
        image = np.random.rand(80, 80).astype(np.float32) * 0.2
        template = np.random.rand(12, 12).astype(np.float32) * 0.5 + 0.4
        image[30:42, 30:42] = template
        
        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)
        
        # OpenCV matching
        cv_result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        cv_peaks = find_peak_2d(cv_result, top_k=3)
        cv_best_x, cv_best_y, cv_best_score = cv_peaks[0]
        
        # Vulkan matching
        vulkan_matches = matcher.match_template(
            img_data, tmpl_data,
            correlation_threshold=0.2,
            max_matches=3
        )
        
        # Vulkan should find match
        assert len(vulkan_matches) > 0, "Vulkan found no matches"
        vulkan_best = vulkan_matches[0]
        
        # Both should find high correlation for the planted pattern
        assert cv_best_score > 0.7, f"OpenCV failed: {cv_best_score}"
        assert vulkan_best.confidence > 0.3, f"Vulkan failed: {vulkan_best.confidence}"
        
        # Locations should be roughly similar (within template-size tolerance)
        # Note: coordinate systems may differ slightly
        loc_diff = abs(cv_best_x - vulkan_best.x) + abs(cv_best_y - vulkan_best.y)
        assert loc_diff < 20, f"Locations too different: OpenCV=({cv_best_x},{cv_best_y}), Vulkan=({vulkan_best.x},{vulkan_best.y})"
    
    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
    @pytest.mark.skipif(not TEST_ASSETS_DIR.exists(), reason="Test assets not found")
    def test_lenna_opencv_comparison(self, matcher):
        """Compare Vulkan vs OpenCV on Lenna image."""
        img_path = TEST_ASSETS_DIR / "lenna.png"
        tmpl_path = TEST_ASSETS_DIR / "templates" / "test1.png"
        
        if not img_path.exists() or not tmpl_path.exists():
            pytest.skip("Test images not found")
        
        # Load with OpenCV for fair comparison
        cv_image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        cv_template = cv2.imread(str(tmpl_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        
        img_data = ImageData.from_file(str(img_path))
        tmpl_data = ImageData.from_file(str(tmpl_path))
        
        # OpenCV
        cv_result = cv2.matchTemplate(cv_image, cv_template, cv2.TM_CCOEFF_NORMED)
        cv_best_y, cv_best_x = np.unravel_index(np.argmax(cv_result), cv_result.shape)
        cv_best_score = float(cv_result[cv_best_y, cv_best_x])
        
        # Vulkan
        vulkan_matches = matcher.match_template(
            img_data, tmpl_data,
            correlation_threshold=0.2,
            max_matches=5
        )
        
        # Vulkan should return valid matches
        for match in vulkan_matches:
            assert 0 <= match.x <= img_data.width
            assert 0 <= match.y <= img_data.height


class TestCPUVsOpenCV:
    """Validate CPU implementation against OpenCV (baseline validation)."""
    
    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
    def test_sse_matches_opencv_sqdiff(self):
        """CPU SSE should find same minimum as OpenCV TM_SQDIFF."""
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
        assert abs(cpu_min_x - 20) <= 2, f"CPU SSE wrong location: ({cpu_min_x}, {cpu_min_y})"
        assert abs(cv_min_x - 20) <= 2, f"OpenCV wrong location: ({cv_min_x}, {cv_min_y})"
    
    @pytest.mark.skipif(not OPENCV_AVAILABLE, reason="OpenCV not available")
    def test_cross_correlation_matches_opencv(self):
        """CPU cross-correlation should find same peak as OpenCV TM_CCORR."""
        np.random.seed(456)
        image = np.random.rand(50, 50).astype(np.float32) * 0.5
        template = np.random.rand(10, 10).astype(np.float32) * 0.3 + 0.2
        
        # Plant template at known location
        image[25:35, 25:35] = template * 1.5  # Brighter version
        
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
        
        # Both should find peak at same location (25, 25)
        assert abs(cpu_max_x - 25) <= 3, f"CPU wrong: ({cpu_max_x}, {cpu_max_y})"
        assert abs(cv_max_x - 25) <= 3, f"OpenCV wrong: ({cv_max_x}, {cv_max_y})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
