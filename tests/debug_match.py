#!/usr/bin/env python3
"""Debug: Show exactly what's happening in template matching."""

import time
import math
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

from vulkan_tensor_matching import ImageData, VulkanTensorMatcher


def extract_patch(img_data, x, y, w, h):
    patch = []
    for row in range(h):
        for col in range(w):
            px, py = x + col, y + row
            if 0 <= py < img_data.height and 0 <= px < img_data.width:
                idx = py * img_data.width + px
                patch.append(img_data.data[idx])
            else:
                patch.append(0.0)
    return patch


def patch_to_image(patch, w, h):
    img = Image.new("L", (w, h))
    for i, val in enumerate(patch):
        img.putpixel((i % w, i // w), int(max(0, min(255, val * 255))))
    return img


def main():
    print("=" * 60)
    print("DEBUG: Template Matching Step-by-Step")
    print("=" * 60)

    lenna_path = Path("test_data/lenna.png")
    lenna = ImageData.from_file(str(lenna_path))
    lenna_pil = Image.open(lenna_path).convert("RGB")

    tmpl_x, tmpl_y = 100, 100
    tmpl_size = 64

    print(f"\nExtracting {tmpl_size}x{tmpl_size} template from ({tmpl_x}, {tmpl_y})...")
    template_data = extract_patch(lenna, tmpl_x, tmpl_y, tmpl_size, tmpl_size)

    tmpl_img = patch_to_image(template_data, tmpl_size, tmpl_size)
    tmpl_path = Path("test_data/debug_template.png")
    tmpl_img.save(tmpl_path)
    print(f"Saved template to {tmpl_path}")

    template = ImageData(template_data, tmpl_size, tmpl_size, 1)

    print("\nMatching template back to image...")
    matcher = VulkanTensorMatcher()
    start = time.time()
    matches = matcher.match_template(lenna, template, 0.3, 5)
    duration = (time.time() - start) * 1000

    print(f"Found {len(matches)} matches in {duration:.1f}ms")

    expected_cx = tmpl_x + tmpl_size // 2
    expected_cy = tmpl_y + tmpl_size // 2

    for i, m in enumerate(matches):
        dx = m.x - expected_cx
        dy = m.y - expected_cy
        dist = math.sqrt(dx * dx + dy * dy)
        deg = math.degrees(m.rotation_angle)
        print(
            f"  Match {i + 1}: ({m.x}, {m.y}) corr={m.correlation:.3f} rot={m.rotation_angle:.2f}rad ({deg:.1f}°) dist={dist:.1f}px"
        )

    draw = ImageDraw.Draw(lenna_pil)
    draw.rectangle(
        [tmpl_x, tmpl_y, tmpl_x + tmpl_size, tmpl_y + tmpl_size],
        outline="purple",
        width=3,
    )
    draw.text((tmpl_x, tmpl_y - 20), f"Expected ({tmpl_x}, {tmpl_y})", fill="purple")

    for i, m in enumerate(matches[:3]):
        det_x = int(m.x - tmpl_size / 2)
        det_y = int(m.y - tmpl_size / 2)
        color = "green" if m.correlation > 0.8 else "orange"
        draw.rectangle(
            [det_x, det_y, det_x + tmpl_size, det_y + tmpl_size], outline=color, width=2
        )
        draw.text((det_x, det_y - 20), f"#{i + 1}: {m.correlation:.2f}", fill=color)

    viz_path = Path("test_data/debug_match.png")
    lenna_pil.save(viz_path)
    print(f"\nSaved visualisation to {viz_path}")

    # NumPy NCC reference
    print("\n" + "=" * 60)
    print("Reference: NumPy NCC")
    print("=" * 60)

    lenna_np = np.array(Image.open(lenna_path).convert("L")).astype(np.float32) / 255.0
    tmpl_np = np.array(tmpl_img).astype(np.float32) / 255.0

    tmpl_mean = np.mean(tmpl_np)
    tmpl_std = np.std(tmpl_np)
    tmpl_norm = (tmpl_np - tmpl_mean) / (tmpl_std * np.sqrt(tmpl_size * tmpl_size))

    best_corr = -1
    best_loc = None

    step = 4
    for y in range(0, lenna_np.shape[0] - tmpl_size, step):
        for x in range(0, lenna_np.shape[1] - tmpl_size, step):
            patch = lenna_np[y : y + tmpl_size, x : x + tmpl_size]
            patch_mean = np.mean(patch)
            patch_std = np.std(patch)
            if patch_std < 1e-6:
                continue
            patch_norm = (patch - patch_mean) / (
                patch_std * np.sqrt(tmpl_size * tmpl_size)
            )
            corr = np.sum(tmpl_norm * patch_norm)
            if corr > best_corr:
                best_corr = corr
                best_loc = (x, y)

    if best_loc:
        best_cx = best_loc[0] + tmpl_size // 2
        best_cy = best_loc[1] + tmpl_size // 2
        dx = best_cx - expected_cx
        dy = best_cy - expected_cy
        dist = math.sqrt(dx * dx + dy * dy)
        print(
            f"Best match: ({best_loc[0]}, {best_loc[1]}) centre=({best_cx}, {best_cy}) corr={best_corr:.3f}"
        )
        print(f"Distance to expected: {dist:.1f}px")
        if dist < 15:
            print("✓ NUMPY NCC CORRECT")
        else:
            print("✗ NUMPY NCC WRONG")


if __name__ == "__main__":
    main()
