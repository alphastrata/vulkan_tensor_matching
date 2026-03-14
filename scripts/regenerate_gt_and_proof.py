#!/usr/bin/env python3
"""
Regenerate Ground Truth and Proof Document

For each source image:
1. Pick a distinctive random location
2. Extract template from that location
3. Save template to test_data/templates/
4. Match with our Vulkan implementation
5. Match with OpenCV NCC (reference)
6. Record correct ground truth
7. Generate PROOF.md with accurate data
"""

import json
import time
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

try:
    import cv2

    OPENCV_AVAILABLE = True
    print("✓ OpenCV available")
except Exception as e:
    print(f"✗ OpenCV not available: {e}")
    OPENCV_AVAILABLE = False

from vulkan_tensor_matching import ImageData, VulkanTensorMatcher

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
TEMPLATES_DIR = TEST_DATA_DIR / "templates"
OUTPUT_DIR = TEST_DATA_DIR / "proof_output"
PROOF_MD = TEST_DATA_DIR / "PROOF.md"
GT_FILE = TEST_DATA_DIR / "answers.jsonl"

# Ensure directories exist
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Template size
TEMPLATE_SIZE = 64


def extract_and_save_template(img_data, x, y, size, output_path):
    """Extract template from image and save."""
    patch = []
    for row in range(size):
        for col in range(size):
            px, py = x + col, y + row
            if 0 <= py < img_data.height and 0 <= px < img_data.width:
                idx = py * img_data.width + px
                patch.append(img_data.data[idx])
            else:
                patch.append(0.0)

    # Save as PIL image
    tmpl_img = Image.new("L", (size, size))
    for i, val in enumerate(patch):
        tmpl_img.putpixel((i % size, i // size), int(max(0, min(255, val * 255))))
    tmpl_img.save(output_path)

    return patch


def find_best_location(img_gray, template_size=64, margin=50):
    """Find a distinctive location for template extraction."""
    ih, iw = img_gray.shape

    # Compute variance in sliding windows
    best_loc = None
    best_var = 0

    step = 20
    for y in range(margin, ih - template_size - margin, step):
        for x in range(margin, iw - template_size - margin, step):
            patch = img_gray[y : y + template_size, x : x + template_size]
            var = np.var(patch)
            if var > best_var:
                best_var = var
                best_loc = (x, y)

    return best_loc


def opencv_match(source_gray, tmpl_gray):
    """Match template using OpenCV NCC."""
    if not OPENCV_AVAILABLE:
        return None

    result = cv2.matchTemplate(source_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    th, tw = tmpl_gray.shape
    centre_x = max_loc[0] + tw / 2
    centre_y = max_loc[1] + th / 2

    return {
        "x": centre_x,
        "y": centre_y,
        "corr": float(max_val),
        "top_left": (max_loc[0], max_loc[1]),
    }


def draw_match(img, x, y, w, h, color, label, is_centre=True):
    """Draw match annotation."""
    draw = ImageDraw.Draw(img, "RGBA")

    if is_centre:
        tl_x, tl_y = x - w / 2, y - h / 2
    else:
        tl_x, tl_y = x, y

    # Box
    draw.rectangle([tl_x, tl_y, tl_x + w, tl_y + h], outline=color, width=3)

    # Cross at centre
    cx, cy = tl_x + w / 2, tl_y + h / 2
    draw.line([(cx - 10, cy), (cx + 10, cy)], fill=color, width=2)
    draw.line([(cx, cy - 10), (cx, cy + 10)], fill=color, width=2)

    # Label
    draw.text((tl_x, tl_y - 20), label, fill=color)


def process_image(img_path, case_idx, matcher):
    """Process single image: extract template, match, generate proof."""
    print(f"\n{'=' * 60}")
    print(f"Case {case_idx}: {img_path.name}")
    print(f"{'=' * 60}")

    # Load image
    img_pil = Image.open(img_path).convert("L")
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_data = ImageData.from_file(str(img_path))

    ih, iw = img_np.shape

    # Find distinctive location
    loc = find_best_location(img_np, TEMPLATE_SIZE)
    if loc is None:
        print(f"  SKIP: Could not find distinctive location")
        return None

    tmpl_x, tmpl_y = loc
    tmpl_centre_x = tmpl_x + TEMPLATE_SIZE // 2
    tmpl_centre_y = tmpl_y + TEMPLATE_SIZE // 2

    print(
        f"  Template location: top-left=({tmpl_x}, {tmpl_y}), centre=({tmpl_centre_x}, {tmpl_centre_y})"
    )

    # Extract and save template
    tmpl_path = TEMPLATES_DIR / f"gt_{case_idx:03d}.png"
    extract_and_save_template(img_data, tmpl_x, tmpl_y, TEMPLATE_SIZE, tmpl_path)
    print(f"  Saved template: {tmpl_path}")

    # Load template for matching
    tmpl_data = ImageData.from_file(str(tmpl_path))
    tmpl_pil = Image.open(tmpl_path).convert("L")
    tmpl_np = np.array(tmpl_pil).astype(np.float32) / 255.0

    # Match with Vulkan
    start = time.time()
    vulkan_matches = matcher.match_template(img_data, tmpl_data, 0.3, 3)
    vulkan_duration = (time.time() - start) * 1000

    vulkan_result = None
    if vulkan_matches:
        best = vulkan_matches[0]
        vulkan_result = {
            "x": best.x,
            "y": best.y,
            "corr": best.correlation,
            "rotation": best.rotation_angle,
        }
        print(
            f"  Vulkan: ({best.x}, {best.y}) corr={best.correlation:.3f} ({vulkan_duration:.1f}ms)"
        )

    # Match with OpenCV
    opencv_result = opencv_match(img_np, tmpl_np)
    if opencv_result:
        print(
            f"  OpenCV: ({opencv_result['x']:.1f}, {opencv_result['y']:.1f}) corr={opencv_result['corr']:.3f}"
        )

    # Compute distances
    if vulkan_result:
        vulkan_dist = (
            (vulkan_result["x"] - tmpl_centre_x) ** 2
            + (vulkan_result["y"] - tmpl_centre_y) ** 2
        ) ** 0.5
        print(f"  Vulkan distance to GT: {vulkan_dist:.1f}px")

    if opencv_result:
        opencv_dist = (
            (opencv_result["x"] - tmpl_centre_x) ** 2
            + (opencv_result["y"] - tmpl_centre_y) ** 2
        ) ** 0.5
        print(f"  OpenCV distance to GT: {opencv_dist:.1f}px")

    # Generate visualisation
    rgb_img = Image.open(img_path).convert("RGB")

    # Ground truth (purple)
    draw_match(
        rgb_img,
        tmpl_centre_x,
        tmpl_centre_y,
        TEMPLATE_SIZE,
        TEMPLATE_SIZE,
        (128, 0, 128),
        "GT",
        is_centre=True,
    )

    # Vulkan match (green)
    if vulkan_result:
        color = (0, 255, 0) if vulkan_dist < 15 else (255, 165, 0)
        draw_match(
            rgb_img,
            vulkan_result["x"],
            vulkan_result["y"],
            TEMPLATE_SIZE,
            TEMPLATE_SIZE,
            color,
            f"Vulkan: {vulkan_result['corr']:.2f}",
            is_centre=True,
        )

    # OpenCV match (orange)
    if opencv_result:
        draw_match(
            rgb_img,
            opencv_result["x"],
            opencv_result["y"],
            TEMPLATE_SIZE,
            TEMPLATE_SIZE,
            (255, 165, 0),
            f"OpenCV: {opencv_result['corr']:.2f}",
            is_centre=True,
        )

    # Save visualisation
    viz_path = OUTPUT_DIR / f"case_{case_idx:03d}_proof.png"
    rgb_img.save(viz_path)

    # Save template visualisation
    tmpl_viz = tmpl_pil.convert("RGB")
    tmpl_viz_path = OUTPUT_DIR / f"case_{case_idx:03d}_template.png"
    tmpl_viz.save(tmpl_viz_path)

    return {
        "image_path": str(img_path.relative_to(TEST_DATA_DIR)),
        "template_path": str(tmpl_path.relative_to(TEST_DATA_DIR)),
        "expected_matches": [
            {
                "x": tmpl_x,
                "y": tmpl_y,
                "w": TEMPLATE_SIZE,
                "h": TEMPLATE_SIZE,
                "angle": 0.0,
            }
        ],
        "vulkan_result": vulkan_result,
        "opencv_result": opencv_result,
        "vulkan_distance": vulkan_dist if vulkan_result else None,
        "opencv_distance": opencv_dist if opencv_result else None,
        "vulkan_duration_ms": vulkan_duration,
        "visualisation": str(viz_path.relative_to(TEST_DATA_DIR)),
        "template_viz": str(tmpl_viz_path.relative_to(TEST_DATA_DIR)),
    }


def generate_proof_md(results):
    """Generate PROOF.md document."""
    md = [
        "# Vulkan Tensorial Template Matching - Visual Proof",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        "This proof document uses **correctly generated ground truth**:",
        "1. For each source image, a distinctive location is automatically selected",
        "2. A 64x64 template is extracted from that location",
        "3. Both Vulkan TTM and OpenCV NCC attempt to find the template",
        "4. Ground truth is the **exact extraction location**",
        "",
        "| Case | Image | Vulkan Corr | OpenCV Corr | Vulkan Dist | OpenCV Dist | Status |",
        "|------|-------|-------------|-------------|-------------|-------------|--------|",
    ]

    for r in results:
        status = (
            "✓"
            if (r["vulkan_distance"] < 15 if r["vulkan_distance"] else False)
            else "⚠️"
        )
        vulkan_corr = (
            f"{r['vulkan_result']['corr']:.3f}" if r["vulkan_result"] else "N/A"
        )
        opencv_corr = (
            f"{r['opencv_result']['corr']:.3f}" if r["opencv_result"] else "N/A"
        )
        vulkan_dist = f"{r['vulkan_distance']:.1f}px" if r["vulkan_distance"] else "N/A"
        opencv_dist = f"{r['opencv_distance']:.1f}px" if r["opencv_distance"] else "N/A"

        md.append(
            f"| {len(md) - 5} | {Path(r['image_path']).name} | {vulkan_corr} | {opencv_corr} | {vulkan_dist} | {opencv_dist} | {status} |"
        )

    md.extend(["", "---", ""])

    # Detailed results per case
    for i, r in enumerate(results):
        md.append(f"### Case {i}: {Path(r['image_path']).name}")
        md.append("")
        md.append(
            f"**Template Location (Ground Truth)**: top-left=({r['expected_matches'][0]['x']}, {r['expected_matches'][0]['y']})"
        )
        md.append("")

        if r["vulkan_result"]:
            md.append(f"**Vulkan TTM Result**:")
            md.append(
                f"- Position: ({r['vulkan_result']['x']:.1f}, {r['vulkan_result']['y']:.1f})"
            )
            md.append(f"- Correlation: {r['vulkan_result']['corr']:.3f}")
            md.append(
                f"- Rotation: {r['vulkan_result']['rotation']:.1f} rad ({np.degrees(r['vulkan_result']['rotation']):.1f}°)"
            )
            md.append(f"- Distance from GT: {r['vulkan_distance']:.1f}px")
            md.append(f"- Duration: {r['vulkan_duration_ms']:.1f}ms")

        if r["opencv_result"]:
            md.append("")
            md.append(f"**OpenCV NCC Result**:")
            md.append(
                f"- Position: ({r['opencv_result']['x']:.1f}, {r['opencv_result']['y']:.1f})"
            )
            md.append(f"- Correlation: {r['opencv_result']['corr']:.3f}")
            md.append(f"- Distance from GT: {r['opencv_distance']:.1f}px")

        md.append("")
        md.append(f"![Proof]({r['visualisation']})")
        md.append("")
        md.append(f"![Template]({r['template_viz']})")
        md.append("")
        md.append("---")
        md.append("")

    # Write file
    with open(PROOF_MD, "w") as f:
        f.write("\n".join(md))

    print(f"\n✓ Generated PROOF.md")


def main():
    print("=" * 60)
    print("Regenerating Ground Truth and Proof Document")
    print("=" * 60)

    # Initialise matcher
    print("\nInitializing Vulkan Tensor Matcher...")
    matcher = VulkanTensorMatcher()

    # Find all source images
    source_dir = TEST_DATA_DIR / "extensive" / "source_images"
    if not source_dir.exists():
        # Use lenna.png as single test
        source_images = [TEST_DATA_DIR / "lenna.png"]
    else:
        source_images = list(source_dir.glob("*.png"))[:10]  # First 10 for speed

    print(f"\nProcessing {len(source_images)} images...")

    results = []
    for i, img_path in enumerate(source_images):
        result = process_image(img_path, i, matcher)
        if result:
            results.append(result)

    # Generate proof document
    print("\nGenerating PROOF.md...")
    generate_proof_md(results)

    # Save ground truth
    gt_data = []
    for r in results:
        gt_data.append(
            {
                "image_path": r["image_path"],
                "template_path": r["template_path"],
                "expected_matches": r["expected_matches"],
            }
        )

    with open(GT_FILE, "w") as f:
        for item in gt_data:
            f.write(json.dumps(item) + "\n")

    print(f"✓ Generated {GT_FILE}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    vulkan_success = sum(
        1 for r in results if r["vulkan_distance"] and r["vulkan_distance"] < 15
    )
    opencv_success = sum(
        1 for r in results if r["opencv_distance"] and r["opencv_distance"] < 15
    )

    print(f"\nOut of {len(results)} cases:")
    print(f"  Vulkan TTM success (< 15px): {vulkan_success}/{len(results)}")
    print(f"  OpenCV NCC success (< 15px): {opencv_success}/{len(results)}")

    if vulkan_success == len(results):
        print("\n✓✓✓ ALL VULKAN MATCHES PERFECT! ✓✓✓")
    elif vulkan_success > len(results) * 0.8:
        print("\n✓ MOST VULKAN MATCHES CORRECT")
    else:
        print("\n⚠ Some Vulkan matches need investigation")

    print(f"\nOutput files:")
    print(f"  - {PROOF_MD}")
    print(f"  - {GT_FILE}")
    print(f"  - {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
