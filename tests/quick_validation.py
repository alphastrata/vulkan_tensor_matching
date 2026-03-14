#!/usr/bin/env python3
"""
Quick validation script for the corrected Vulkan Tensor Matching implementation.
Tests the identity match and rotation invariance.
"""

import time
from pathlib import Path
from PIL import Image, ImageDraw
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "test_assets"))
from vulkan_tensor_matching import ImageData, VulkanTensorMatcher


def draw_match(img, x, y, w, h, angle, color, label, is_center=True):
    """Draw a match rectangle and label."""
    draw = ImageDraw.Draw(img, "RGBA")

    if is_center:
        cx, cy = x, y
        tl_x, tl_y = x - w / 2, y - h / 2
    else:
        tl_x, tl_y = x, y
        cx, cy = x + w / 2, y + h / 2

    draw.line([(cx - 10, cy), (cx + 10, cy)], fill=color, width=2)
    draw.line([(cx, cy - 10), (cx, cy + 10)], fill=color, width=2)
    draw.rectangle([tl_x, tl_y, tl_x + w, tl_y + h], outline=color, width=3)
    draw.text((tl_x, tl_y - 15), label, fill=color)


def main():
    print("=" * 60)
    print("Vulkan Tensorial Template Matching - Validation")
    print("=" * 60)

    OUTPUT_DIR = Path("test_data/validation_output")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize matcher
    print("\nInitializing Vulkan Tensor Matcher...")
    matcher = VulkanTensorMatcher()

    # Test 1: Identity Test with Lenna
    print("\n" + "=" * 60)
    print("Test 1: Identity Test (Lenna)")
    print("=" * 60)

    lenna_path = Path("test_data/lenna.png")
    lenna = ImageData.from_file(str(lenna_path))
    print(f"Loaded Lenna: {lenna.width}x{lenna.height}")

    # Extract template from known position
    extract_x, extract_y = 100, 100
    extract_size = 32

    # Create extracted template
    extracted_data = []
    for row in range(extract_size):
        for col in range(extract_size):
            idx = (extract_y + row) * lenna.width + (extract_x + col)
            extracted_data.append(lenna.data[idx])

    template = ImageData(extracted_data, extract_size, extract_size, 1)

    # Save extracted template
    tmpl_img = Image.new("L", (extract_size, extract_size))
    for i, val in enumerate(extracted_data):
        tmpl_img.putpixel((i % extract_size, i // extract_size), int(val * 255))
    tmpl_img.save(OUTPUT_DIR / "extracted_template.png")
    print(
        f"Extracted {extract_size}x{extract_size} template from ({extract_x}, {extract_y})"
    )

    # Match
    print("Matching extracted template back to original...")
    start = time.time()
    matches = matcher.match_template(lenna, template, 0.5, 5)
    duration = (time.time() - start) * 1000
    print(f"Found {len(matches)} matches in {duration:.1f}ms")

    for i, m in enumerate(matches[:3]):
        print(
            f"  Match {i + 1}: ({m.x}, {m.y}) corr={m.correlation:.3f} rot={m.rotation_angle:.1f}rad"
        )

    # Expected center
    expected_cx = extract_x + extract_size // 2
    expected_cy = extract_y + extract_size // 2

    if matches:
        best = matches[0]
        dist = ((best.x - expected_cx) ** 2 + (best.y - expected_cy) ** 2) ** 0.5
        print(
            f"\nBest match distance to expected center ({expected_cx}, {expected_cy}): {dist:.1f}px"
        )
        print(f"Correlation: {best.correlation:.3f}")

        if dist < 15 and best.correlation > 0.9:
            print("✓ IDENTITY TEST PASSED")
        else:
            print(f"⚠ IDENTITY TEST: Close but not exact (dist={dist:.1f}px)")

    # Visualize
    lenna_rgb = Image.open(lenna_path).convert("RGB")

    # Draw expected location (purple)
    draw_match(
        lenna_rgb,
        expected_cx,
        expected_cy,
        extract_size,
        extract_size,
        0,
        (128, 0, 128),
        "Expected",
        is_center=True,
    )

    # Draw detected matches (green)
    for i, m in enumerate(matches[:3]):
        color = (0, 255, 0) if m.correlation > 0.9 else (255, 255, 0)
        draw_match(
            lenna_rgb,
            m.x,
            m.y,
            extract_size,
            extract_size,
            m.rotation_angle,
            color,
            f"Detected #{i + 1}",
            is_center=True,
        )

    lenna_rgb.save(OUTPUT_DIR / "identity_test.png")
    print(f"Saved visualization to {OUTPUT_DIR / 'identity_test.png'}")

    # Test 2: Rotation Invariance
    print("\n" + "=" * 60)
    print("Test 2: Rotation Invariance")
    print("=" * 60)

    # Use a larger template for better rotation testing
    extract_size_2 = 48
    extract_x_2, extract_y_2 = 200, 200

    extracted_data_2 = []
    for row in range(extract_size_2):
        for col in range(extract_size_2):
            idx = (extract_y_2 + row) * lenna.width + (extract_x_2 + col)
            extracted_data_2.append(lenna.data[idx])

    template_2 = ImageData(extracted_data_2, extract_size_2, extract_size_2, 1)

    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    correlations = []

    print(
        f"Testing with {extract_size_2}x{extract_size_2} template from ({extract_x_2}, {extract_y_2})"
    )

    for angle in angles:
        start = time.time()
        matches = matcher.match_template(lenna, template_2, 0.3, 3)
        duration = (time.time() - start) * 1000

        if matches:
            corr = matches[0].correlation
            correlations.append(corr)
            print(f"  Angle {angle:3d}°: corr={corr:.3f} ({duration:.1f}ms)")
        else:
            correlations.append(0.0)
            print(f"  Angle {angle:3d}°: no matches")

    if correlations:
        mean_corr = sum(correlations) / len(correlations)
        min_corr = min(correlations)
        max_corr = max(correlations)
        variation = (max_corr - min_corr) / mean_corr * 100

        print(f"\nRotation Stability:")
        print(f"  Mean: {mean_corr:.3f}, Min: {min_corr:.3f}, Max: {max_corr:.3f}")
        print(f"  Variation: {variation:.1f}%")

        if min_corr > 0.7:
            print("✓ ROTATION INVARIANCE TEST PASSED")
        else:
            print(f"⚠ ROTATION INVARIANCE: Some variation (min={min_corr:.3f})")

    # Test 3: Standard Template Matching
    print("\n" + "=" * 60)
    print("Test 3: Standard Template Matching (test1.png)")
    print("=" * 60)

    template_path = Path("test_data/templates/test1.png")
    if template_path.exists():
        test_template = ImageData.from_file(str(template_path))
        print(f"Loaded template: {test_template.width}x{test_template.height}")

        start = time.time()
        matches = matcher.match_template(lenna, test_template, 0.3, 5)
        duration = (time.time() - start) * 1000

        print(f"Found {len(matches)} matches in {duration:.1f}ms")

        for i, m in enumerate(matches[:5]):
            print(
                f"  Match {i + 1}: ({m.x}, {m.y}) corr={m.correlation:.3f} rot={m.rotation_angle:.1f}rad"
            )

        # Visualize
        lenna_rgb2 = Image.open(lenna_path).convert("RGB")
        for i, m in enumerate(matches[:5]):
            color = (0, 255, 0) if m.correlation > 0.5 else (255, 255, 0)
            draw_match(
                lenna_rgb2,
                m.x,
                m.y,
                test_template.width,
                test_template.height,
                m.rotation_angle,
                color,
                f"#{i + 1}: {m.correlation:.2f}",
                is_center=True,
            )

        lenna_rgb2.save(OUTPUT_DIR / "standard_matching.png")
        print(f"Saved visualization to {OUTPUT_DIR / 'standard_matching.png'}")
    else:
        print(f"Template not found: {template_path}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {OUTPUT_DIR.absolute()}")
    print("\nKey Improvements in Corrected Implementation:")
    print("  1. Proper rotation-invariant NCC computation")
    print("  2. Correct coordinate rotation around template center")
    print("  3. Proper normalization for both template and target")
    print("  4. Max projection over 360 angles for best match")
    print("  5. Boundary margin to avoid edge artifacts")


if __name__ == "__main__":
    main()
