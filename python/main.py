#!/usr/bin/env python3
"""
Example usage of the vulkan_tensor_matching Python bindings.

Demonstrates CPU and GPU-accelerated template matching with comparison to OpenCV.
"""

import numpy as np
from pathlib import Path

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("OpenCV not available - some examples will be skipped")

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


def image_data_to_numpy(image_data: ImageData) -> np.ndarray:
    """Convert ImageData to numpy array."""
    return np.array(image_data.data, dtype=np.float32).reshape(
        (image_data.height, image_data.width)
    )


def numpy_to_image_data(arr: np.ndarray) -> ImageData:
    """Convert numpy array to ImageData."""
    arr = arr.astype(np.float32)
    return ImageData(
        data=arr.flatten().tolist(),
        width=arr.shape[1],
        height=arr.shape[0],
        channels=1,
    )


def main():
    print(f"vulkan_tensor_matching Python bindings v{VERSION}")
    print(f"Author: {AUTHOR}")
    print("=" * 60)
    
    # Example 1: Basic ImageData operations
    print("\n1. ImageData operations:")
    print("-" * 40)
    
    # Create a simple test image
    test_image = np.zeros((50, 50), dtype=np.float32)
    test_image[20:30, 20:30] = 1.0  # Bright square in center
    
    img = numpy_to_image_data(test_image)
    print(f"   Created image: {img}")
    
    # Find extremes
    extremes = find_extremes(img)
    print(f"   Max value: {extremes['max']['value']:.4f} at ({extremes['max']['x']}, {extremes['max']['y']})")
    print(f"   Min value: {extremes['min']['value']:.4f} at ({extremes['min']['x']}, {extremes['min']['y']})")
    
    # Compress image
    compressed = compress_image(img, factor=2)
    print(f"   Compressed: {compressed.width}x{compressed.height} (from {img.width}x{img.height})")
    
    # Example 2: CPU template matching
    print("\n2. CPU Template Matching:")
    print("-" * 40)
    
    # Create template from a region of the image
    template = test_image[20:30, 20:30].copy()
    
    img_data = numpy_to_image_data(test_image)
    tmpl_data = numpy_to_image_data(template)
    
    # Run template matching
    result = match_template_cpu(
        img_data, tmpl_data,
        MatchTemplateMethod.cross_correlation_normalized()
    )
    result_arr = image_data_to_numpy(result)
    
    # Find best match location
    best_y, best_x = np.unravel_index(np.argmax(result_arr), result_arr.shape)
    best_score = result_arr[best_y, best_x]
    
    print(f"   Template size: {template.shape[0]}x{template.shape[1]}")
    print(f"   Best match at: ({best_x}, {best_y}) with score: {best_score:.4f}")
    print(f"   Expected location: around (25, 25) - center of bright square")
    
    # Example 3: Compare with OpenCV
    if OPENCV_AVAILABLE:
        print("\n3. OpenCV Comparison:")
        print("-" * 40)
        
        cv_result = cv2.matchTemplate(test_image, template, cv2.TM_CCOEFF_NORMED)
        cv_best_y, cv_best_x = np.unravel_index(np.argmax(cv_result), cv_result.shape)
        cv_best_score = cv_result[cv_best_y, cv_best_x]
        
        print(f"   OpenCV best match: ({cv_best_x}, {cv_best_y}) with score: {cv_best_score:.4f}")
        print(f"   Rust best match:   ({best_x}, {best_y}) with score: {best_score:.4f}")
        
        # Check if both found similar locations
        distance = np.sqrt((best_x - cv_best_x)**2 + (best_y - cv_best_y)**2)
        print(f"   Location difference: {distance:.2f} pixels")
        
        if distance < 5:
            print("   ✓ Both implementations found similar match locations!")
        else:
            print("   ⚠ Match locations differ (may be due to different algorithms)")
    else:
        print("\n3. OpenCV Comparison: SKIPPED (OpenCV not installed)")
    
    # Example 4: GPU-accelerated matching (if Vulkan available)
    print("\n4. GPU-accelerated Template Matching (Vulkan):")
    print("-" * 40)
    
    try:
        matcher = VulkanTensorMatcher()
        print("   Vulkan matcher initialized successfully!")
        
        # Try matching with a threshold
        matches = matcher.match_template(
            img_data, tmpl_data,
            correlation_threshold=0.5,
            max_matches=5
        )
        
        print(f"   Found {len(matches)} matches above threshold:")
        for i, match in enumerate(matches[:3]):  # Show top 3
            print(f"     {i+1}. ({match.x}, {match.y}) - correlation: {match.correlation:.4f}")
            
    except ValueError as e:
        print(f"   Vulkan not available: {e}")
        print("   (This is expected on systems without Vulkan support)")
    
    # Example 5: Load image from file
    print("\n5. Loading Image from File:")
    print("-" * 40)
    
    test_assets = Path(__file__).parent.parent / "test_assets"
    lenna_path = test_assets / "lenna.png"
    
    if lenna_path.exists():
        lenna = ImageData.from_file(str(lenna_path))
        print(f"   Loaded Lenna: {lenna.width}x{lenna.height}")
        
        # Find extremes in Lenna
        lenna_extremes = find_extremes(lenna)
        print(f"   Max intensity: {lenna_extremes['max']['value']:.4f}")
        print(f"   Min intensity: {lenna_extremes['min']['value']:.4f}")
    else:
        print(f"   Lenna image not found at {lenna_path}")
        print("   (This is expected if test assets are not installed)")
    
    print("\n" + "=" * 60)
    print("Examples completed!")


if __name__ == "__main__":
    main()
