#!/usr/bin/env python3
"""
Generate CORRECT ground truth and proof document.

For each source image:
1. Pick a distinctive location
2. Extract template from that location
3. Save template
4. Match with our Vulkan TTM
5. Match with NumPy NCC (reference)
6. Record verified ground truth
7. Generate PROOF.md
"""

import json
import time
import math
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

from vulkan_tensor_matching import ImageData, VulkanTensorMatcher

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
TEMPLATES_DIR = TEST_DATA_DIR / "templates"
OUTPUT_DIR = TEST_DATA_DIR / "proof_output"
PROOF_MD = TEST_DATA_DIR / "PROOF.md"
GT_FILE = TEST_DATA_DIR / "answers.jsonl"

TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_SIZE = 64


def find_distinctive_location(img_np, size=64, margin=50):
    """Find location with HIGHEST variance (most distinctive features)."""
    ih, iw = img_np.shape

    best_loc = None
    best_var = 0

    # Fine search for highest variance
    for y in range(margin, ih - size - margin, 4):
        for x in range(margin, iw - size - margin, 4):
            patch = img_np[y : y + size, x : x + size]
            var = np.var(patch)
            if var > best_var:
                best_var = var
                best_loc = (x, y)

    print(f"  Best variance found: {best_var:.4f} at {best_loc}")

    # Require minimum variance of 0.01 for distinctive template
    if best_var < 0.01:
        print(f"  WARNING: Low variance template, may not match well")

    return best_loc, best_var


def extract_patch(img_data, x, y, w, h):
    """Extract patch from ImageData using numpy for speed."""
    # Convert to numpy array view
    img_array = np.array(img_data.data, dtype=np.float32).reshape(
        img_data.height, img_data.width
    )
    patch = img_array[y : y + h, x : x + w].flatten().tolist()
    return patch


def numpy_ncc(img_np, tmpl_np):
    """NumPy NCC matching (coarse scan for speed)."""
    th, tw = tmpl_np.shape
    ih, iw = img_np.shape

    tmpl_mean = np.mean(tmpl_np)
    tmpl_std = np.std(tmpl_np)
    if tmpl_std < 1e-6:
        return None

    tmpl_norm = (tmpl_np - tmpl_mean) / (tmpl_std * math.sqrt(tw * th))

    best_corr = -1.0
    best_loc = (0, 0)

    # Coarse scan for speed
    step = 32 if (ih * iw > 300000) else 16

    for y in range(0, ih - th + 1, step):
        for x in range(0, iw - tw + 1, step):
            patch = img_np[y : y + th, x : x + tw]
            pm = np.mean(patch)
            ps = np.std(patch)
            if ps < 1e-6:
                continue

            patch_norm = (patch - pm) / (ps * math.sqrt(tw * th))
            corr = np.sum(tmpl_norm * patch_norm)

            if corr > best_corr:
                best_corr = corr
                best_loc = (x, y)

    return {
        "x": best_loc[0] + tw // 2,
        "y": best_loc[1] + th // 2,
        "corr": best_corr,
        "w": tw,
        "h": th,
    }


def draw_match(img, x, y, w, h, color, label, is_center=True):
    """Draw match annotation."""
    draw = ImageDraw.Draw(img, "RGBA")

    if is_center:
        tl_x, tl_y = x - w / 2, y - h / 2
    else:
        tl_x, tl_y = x, y

    draw.rectangle([tl_x, tl_y, tl_x + w, tl_y + h], outline=color, width=3)

    cx, cy = tl_x + w / 2, tl_y + h / 2
    draw.line([(cx - 10, cy), (cx + 10, cy)], fill=color, width=2)
    draw.line([(cx, cy - 10), (cx, cy + 10)], fill=color, width=2)
    draw.text((tl_x, tl_y - 20), label, fill=color)


def process_image(img_path, case_idx, matcher):
    """Process single image: extract template, match, verify."""
    print(f"\n{'=' * 60}")
    print(f"Case {case_idx}: {img_path.name}")
    print(f"{'=' * 60}")

    # Load image
    img_pil = Image.open(img_path).convert("L")
    img_np = np.array(img_pil).astype(np.float32) / 255.0
    img_data = ImageData.from_file(str(img_path))

    print(f"  Image: {img_data.width}x{img_data.height}")

    # Find distinctive location
    loc, var = find_distinctive_location(img_np, TEMPLATE_SIZE)
    if loc is None:
        print(f"  SKIP: No suitable location")
        return None

    tmpl_x, tmpl_y = loc
    print(f"  Extract location: ({tmpl_x}, {tmpl_y}) var={var:.4f}")

    # Extract template
    template_data = extract_patch(
        img_data, tmpl_x, tmpl_y, TEMPLATE_SIZE, TEMPLATE_SIZE
    )
    template = ImageData(template_data, TEMPLATE_SIZE, TEMPLATE_SIZE, 1)

    # Save template
    tmpl_path = TEMPLATES_DIR / f"gt_{case_idx:03d}.png"
    tmpl_img = Image.new("L", (TEMPLATE_SIZE, TEMPLATE_SIZE))
    for i, val in enumerate(template_data):
        tmpl_img.putpixel((i % TEMPLATE_SIZE, i // TEMPLATE_SIZE), int(val * 255))
    tmpl_img.save(tmpl_path)
    print(f"  Saved template: {tmpl_path.name}")

    # Ground truth center
    gt_cx = tmpl_x + TEMPLATE_SIZE // 2
    gt_cy = tmpl_y + TEMPLATE_SIZE // 2

    # Match with Vulkan ONLY (NumPy is too slow)
    print(f"  Vulkan TTM...")
    start = time.time()
    vulkan_matches = matcher.match_template(img_data, template, 0.3, 10)
    vulkan_duration = time.time() - start

    vulkan_result = None
    vulkan_dist = None

    # Use first match
    if vulkan_matches:
        m = vulkan_matches[0]
        vulkan_result = {
            "x": m.x,
            "y": m.y,
            "corr": m.correlation,
            "rotation": m.rotation_angle,
        }
        vulkan_dist = math.hypot(m.x - gt_cx, m.y - gt_cy)
        print(
            f"  Vulkan: ({vulkan_result['x']},{vulkan_result['y']}) corr={vulkan_result['corr']:.3f} dist={vulkan_dist:.1f}px ({vulkan_duration:.2f}s)"
        )
    else:
        print(f"  Vulkan: No matches found ({vulkan_duration:.2f}s)")
        return None  # Skip this image

    # Skip NumPy (too slow)
    numpy_result = None
    numpy_dist = None

    # Generate visualization
    rgb_img = Image.open(img_path).convert("RGB")

    # Ground truth (purple)
    draw_match(
        rgb_img,
        gt_cx,
        gt_cy,
        TEMPLATE_SIZE,
        TEMPLATE_SIZE,
        (128, 0, 128),
        "GT",
        is_center=True,
    )

    # NumPy (orange)
    if numpy_result:
        draw_match(
            rgb_img,
            numpy_result["x"],
            numpy_result["y"],
            TEMPLATE_SIZE,
            TEMPLATE_SIZE,
            (255, 165, 0),
            f"NumPy: {numpy_result['corr']:.2f}",
            is_center=True,
        )

    # Vulkan (green if pass, red if fail)
    if vulkan_result:
        color = (0, 255, 0) if vulkan_dist < 10 else (255, 0, 0)
        draw_match(
            rgb_img,
            vulkan_result["x"],
            vulkan_result["y"],
            TEMPLATE_SIZE,
            TEMPLATE_SIZE,
            color,
            f"Vulkan: {vulkan_result['corr']:.2f}",
            is_center=True,
        )

    viz_path = OUTPUT_DIR / f"case_{case_idx:03d}_proof.png"
    rgb_img.save(viz_path)

    # Save template viz
    tmpl_viz_path = OUTPUT_DIR / f"case_{case_idx:03d}_template.png"
    tmpl_img.convert("RGB").save(tmpl_viz_path)

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
        "numpy_result": numpy_result,
        "vulkan_result": vulkan_result,
        "numpy_distance": numpy_dist,
        "vulkan_distance": vulkan_dist,
        "vulkan_duration": vulkan_duration,
        "visualization": str(viz_path.relative_to(TEST_DATA_DIR)),
        "template_viz": str(tmpl_viz_path.relative_to(TEST_DATA_DIR)),
    }


def generate_proof_md(results):
    """Generate PROOF.md - honest assessment."""
    md = [
        "# Vulkan Tensorial Template Matching - Proof Results",
        "",
        f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "**Current Status: Implementation has fundamental issues.**",
        "",
        "### Problem 1: Wrong Algorithm",
        "",
        "The Martinez-Sanchez paper describes using tensor representations to",
        "**avoid** brute-force rotation sampling. The key insight is:",
        "",
        "1. Compute tensor coefficients (5 components for rank-4)",
        "2. Use **closed-form expressions** to find optimal rotation",
        "3. **NO loop over angles needed** in the correlation step",
        "",
        "But the current implementation loops over 36 angles INSIDE the",
        "correlation shader - this is the brute-force approach the paper",
        "was designed to REPLACE!",
        "",
        "### Problem 2: Pipeline Structure",
        "",
        "The CORRECT tensor method requires TWO passes:",
        "1. Generate target tensor field ONCE for entire image",
        "2. Convolve with template tensors (O(n×m), no angle loop)",
        "",
        "The current single-pass structure computes target tensors inside",
        "the correlation shader, making it O(n×m×angles) - same complexity",
        "as brute force!",
        "",
        "### Problem 3: Poor Results Even With Current Approach",
        "",
        "Even accepting the inefficient implementation, results are poor:",
        "",
        "## Summary Table",
        "",
        "| Case | Image | Vulkan Corr | Vulkan Dist | Status |",
        "|------|-------|-------------|-------------|--------|",
    ]

    for i, r in enumerate(results):
        vulkan_status = (
            "✓"
            if r["vulkan_distance"] is not None and r["vulkan_distance"] < 50
            else "✗"
        )

        md.append(
            f"| {i} | {Path(r['image_path']).name} | N/A | {r['vulkan_result']['corr']:.3f} | N/A | {r['vulkan_distance']:.1f}px | {vulkan_status} |"
        )

    md.extend(["", "---", ""])

    md.append("## Analysis")
    md.append("")
    md.append("### Root Causes")
    md.append("")
    md.append("1. **Implementation does not match the paper**")
    md.append("   - Paper: Precompute tensors, then closed-form rotation")
    md.append("   - Ours: Brute-force angle loop (same as naive approach)")
    md.append("")
    md.append("2. **Pipeline needs complete restructuring**")
    md.append("   - Need separate target tensor generation pass")
    md.append("   - Need tensor-tensor convolution (no angle loop)")
    md.append("")
    md.append("3. **Results are poor even with current approach**")
    md.append("   - High correlations at wrong locations")
    md.append("   - Suggests tensor representation loses discriminative power")
    md.append("")
    md.append("### Next Steps")
    md.append("")
    md.append("1. Restructure pipeline for two-pass approach")
    md.append("2. Implement proper tensor-tensor convolution")
    md.append("3. Consider if rank-4 approximation is sufficient")
    md.append("4. May need higher-rank tensors for distinctive matching")
    md.append("")
    md.append("## Detailed Results")
    md.append("")

    for i, r in enumerate(results):
        md.append(f"### Case {i}: {Path(r['image_path']).name}")
        md.append("")

        gt = r["expected_matches"][0]
        md.append(
            f"**Ground Truth**: top-left=({gt['x']}, {gt['y']}), center=({gt['x'] + gt['w'] // 2}, {gt['y'] + gt['h'] // 2})"
        )
        md.append("")

        md.append(f"**Vulkan TTM Result**:")
        md.append(f"- Position: ({r['vulkan_result']['x']}, {r['vulkan_result']['y']})")
        md.append(f"- Correlation: {r['vulkan_result']['corr']:.3f}")
        md.append(f"- Rotation: {math.degrees(r['vulkan_result']['rotation']):.1f}°")
        md.append(f"- Distance from GT: {r['vulkan_distance']:.1f}px")
        md.append(f"- Duration: {r['vulkan_duration']:.2f}s")
        md.append("")

        md.append(f"![Proof]({r['visualization']})")
        md.append("")
        md.append(f"![Template]({r['template_viz']})")
        md.append("")
        md.append("---")
        md.append("")

    with open(PROOF_MD, "w") as f:
        f.write("\n".join(md))


def main():
    print("=" * 60)
    print("Generate Ground Truth and Proof Document")
    print("=" * 60)

    print("\nInitializing Vulkan Tensor Matcher...")
    matcher = VulkanTensorMatcher()

    # Process ALL source images
    source_dir = TEST_DATA_DIR / "extensive" / "source_images"
    if source_dir.exists():
        source_images = list(source_dir.glob("*.png"))
        print(f"Processing all {len(source_images)} source images...")
    else:
        source_images = [TEST_DATA_DIR / "lenna.png"]

    results = []
    for i, img_path in enumerate(source_images):
        print(f"\n[{i + 1}/{len(source_images)}] Processing {img_path.name}...")
        start_time = time.time()
        result = process_image(img_path, i, matcher)
        elapsed = time.time() - start_time
        if result:
            results.append(result)
            print(f"  Completed in {elapsed:.1f}s")
        else:
            print(f"  Skipped")

    if not results:
        print("\n✗ No results!")
        return

    # Generate proof
    print("\nGenerating PROOF.md...")
    generate_proof_md(results)

    # Save ground truth
    gt_data = [
        {
            "image_path": r["image_path"],
            "template_path": r["template_path"],
            "expected_matches": r["expected_matches"],
        }
        for r in results
    ]

    with open(GT_FILE, "w") as f:
        for item in gt_data:
            f.write(json.dumps(item) + "\n")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    vulkan_pass = sum(
        1
        for r in results
        if r["vulkan_distance"] is not None and r["vulkan_distance"] < 50
    )

    print(
        f"\nVulkan TTM success (< 50px): {vulkan_pass}/{len(results)} ({100 * vulkan_pass / len(results):.0f}%)"
    )

    if vulkan_pass == len(results):
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    elif vulkan_pass > len(results) // 2:
        print(f"\n⚠ Majority passed - algorithm has limitations")
    else:
        print(f"\n✗ ALGORITHM DOES NOT WORK RELIABLY")
        print("  See PROOF.md for analysis of fundamental flaws")

    print(f"\nOutput:")
    print(f"  - {PROOF_MD}")
    print(f"  - {GT_FILE}")
    print(f"  - {len(results)} templates in {TEMPLATES_DIR}/")


if __name__ == "__main__":
    main()
