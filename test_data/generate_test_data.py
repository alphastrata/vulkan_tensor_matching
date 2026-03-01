#!/usr/bin/env python3
"""
Generate test data for template matching tests.

Creates test images with templates at known locations.
"""

import numpy as np
from pathlib import Path
import cv2

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


def create_test_case(name: str, image_size: tuple, template_size: tuple, 
                     template_positions: list[tuple], noise_level: float = 0.05):
    """
    Create a test case with known template locations.
    
    Args:
        name: Test case name
        image_size: (height, width) of the output image
        template_size: (height, width) of the template
        template_positions: List of (y, x) positions where template should be found
        noise_level: Amount of noise to add to the image
    """
    np.random.seed(42)  # Reproducibility
    
    # Create base image with noise
    image = np.random.rand(*image_size).astype(np.float32) * noise_level
    
    # Create template with distinct pattern
    template = np.zeros(template_size, dtype=np.float32)
    
    # Add a distinctive pattern to the template (gradient + shape)
    cy, cx = template_size[0] // 2, template_size[1] // 2
    for y in range(template_size[0]):
        for x in range(template_size[1]):
            # Radial gradient from center
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            max_dist = np.sqrt(cy**2 + cx**2)
            template[y, x] = 1.0 - (dist / max_dist)
    
    # Add a bright square in the center
    sq_size = min(template_size) // 3
    template[cy-sq_size//2:cy+sq_size//2, cx-sq_size//2:cx+sq_size//2] = 1.0
    
    # Plant template at specified locations
    for ty, tx in template_positions:
        image[ty:ty+template_size[0], tx:tx+template_size[1]] = template
    
    # Save files
    test_dir = TEST_DATA_DIR / name
    test_dir.mkdir(exist_ok=True)
    
    # Save image (as 8-bit PNG)
    image_8bit = (image * 255).astype(np.uint8)
    cv2.imwrite(str(test_dir / "source.png"), image_8bit)
    
    # Save template
    template_8bit = (template * 255).astype(np.uint8)
    cv2.imwrite(str(test_dir / "template.png"), template_8bit)
    
    # Save expected answers (format: x,y,correlation_threshold)
    # For exact matches, correlation should be very high (>0.95)
    with open(test_dir / "expected.txt", "w") as f:
        for ty, tx in template_positions:
            # Center of the template match
            center_x = tx + template_size[1] // 2
            center_y = ty + template_size[0] // 2
            f.write(f"{center_x},{center_y},0.9\n")
    
    print(f"Created test case '{name}':")
    print(f"  Image: {image_size[1]}x{image_size[0]}")
    print(f"  Template: {template_size[1]}x{template_size[0]}")
    print(f"  Expected matches at: {template_positions}")
    print(f"  Output: {test_dir}/")


def main():
    # Test case 1: Single clear match
    create_test_case(
        name="t1_single_match",
        image_size=(200, 200),
        template_size=(30, 30),
        template_positions=[(85, 85)],  # Center of image
        noise_level=0.1
    )
    
    # Test case 2: Multiple matches
    create_test_case(
        name="t2_multi_match",
        image_size=(300, 300),
        template_size=(25, 25),
        template_positions=[(50, 50), (150, 50), (100, 200)],
        noise_level=0.15
    )
    
    # Test case 3: Edge case - match near corner
    create_test_case(
        name="t3_corner_match",
        image_size=(150, 150),
        template_size=(20, 20),
        template_positions=[(10, 10), (120, 120)],  # Near corners
        noise_level=0.1
    )
    
    print("\nTest data generation complete!")


if __name__ == "__main__":
    main()
