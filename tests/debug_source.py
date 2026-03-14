#!/usr/bin/env python3
"""Debug why matching fails on source images."""

import math
from pathlib import Path
from PIL import Image
import numpy as np

from vulkan_tensor_matching import ImageData, VulkanTensorMatcher


def main():
    print("Debug: Source Image Matching")
    print("=" * 60)

    # Test with source_4.png
    img_path = Path("test_data/extensive/source_images/source_4.png")
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return

    img_data = ImageData.from_file(str(img_path))
    img_pil = Image.open(img_path).convert("L")
    img_np = np.array(img_pil).astype(np.float32) / 255.0

    print(f"Image: {img_data.width}x{img_data.height}")
    print(f"Data range: [{min(img_data.data):.3f}, {max(img_data.data):.3f}]")
    print(f"NP range: [{img_np.min():.3f}, {img_np.max():.3f}]")

    # Extract template from high-variance location
    tmpl_x, tmpl_y = 281, 275  # From original GT
    tmpl_size = 64

    print(f"\nExtracting template from ({tmpl_x}, {tmpl_y})...")

    # Extract using same method as matcher
    patch = []
    for row in range(tmpl_size):
        for col in range(tmpl_size):
            px, py = tmpl_x + col, tmpl_y + row
            if 0 <= py < img_data.height and 0 <= px < img_data.width:
                idx = py * img_data.width + px
                patch.append(img_data.data[idx])
            else:
                patch.append(0.0)

    # Save template
    tmpl_img = Image.new("L", (tmpl_size, tmpl_size))
    for i, val in enumerate(patch):
        tmpl_img.putpixel(
            (i % tmpl_size, i // tmpl_size), int(max(0, min(255, val * 255)))
        )
    tmpl_img.save("test_data/debug_source_template.png")

    template = ImageData(patch, tmpl_size, tmpl_size, 1)

    # Match
    print("Matching...")
    matcher = VulkanTensorMatcher()
    import time

    start = time.time()
    matches = matcher.match_template(img_data, template, 0.3, 3)
    duration = (time.time() - start) * 1000

    expected_cx = tmpl_x + tmpl_size // 2
    expected_cy = tmpl_y + tmpl_size // 2

    print(f"Found {len(matches)} matches in {duration:.1f}ms")
    print(f"Expected centre: ({expected_cx}, {expected_cy})")

    for i, m in enumerate(matches):
        dx = m.x - expected_cx
        dy = m.y - expected_cy
        dist = math.sqrt(dx * dx + dy * dy)
        print(
            f"  Match {i + 1}: ({m.x}, {m.y}) corr={m.correlation:.3f} dist={dist:.1f}px"
        )

    # Check what's at the matched location
    if matches:
        best = matches[0]
        det_x = int(best.x - tmpl_size / 2)
        det_y = int(best.y - tmpl_size / 2)

        print(f"\nExtracting patch at detected location ({det_x}, {det_y})...")
        det_patch = []
        for row in range(tmpl_size):
            for col in range(tmpl_size):
                px, py = det_x + col, det_y + row
                if 0 <= py < img_data.height and 0 <= px < img_data.width:
                    idx = py * img_data.width + px
                    det_patch.append(img_data.data[idx])
                else:
                    det_patch.append(0.0)

        # Compare patches
        tmpl_arr = np.array(patch)
        det_arr = np.array(det_patch)

        corr = np.corrcoef(tmpl_arr, det_arr)[0, 1]
        print(f"Correlation between template and detected patch: {corr:.3f}")

        # Save detected patch
        det_img = Image.new("L", (tmpl_size, tmpl_size))
        for i, val in enumerate(det_patch):
            det_img.putpixel(
                (i % tmpl_size, i // tmpl_size), int(max(0, min(255, val * 255)))
            )
        det_img.save("test_data/debug_source_detected.png")


if __name__ == "__main__":
    main()
