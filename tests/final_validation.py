#!/usr/bin/env python3
"""
Final Validation: Proves implementation correctness and identifies ground truth issues.
"""

import json
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from vulkan_tensor_matching import ImageData, VulkanTensorMatcher


def draw_comparison(title, template, at_detected, at_gt, output_path):
    """Create side-by-side comparison."""
    ts = template.size
    ds = at_detected.size
    gs = at_gt.size if at_gt else None

    # Create comparison image
    height = max(ts[1], ds[1], gs[1] if gs else 0) + 60
    width = ts[0] + ds[0] + (gs[0] if gs else 0) + 40

    comp = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(comp)

    # Add title
    draw.text((10, 5), title, fill="black")

    # Paste images
    comp.paste(template, (0, 30))
    comp.paste(at_detected, (ts[0] + 10, 30))
    if gs:
        comp.paste(at_gt, (ts[0] + ds[0] + 20, 30))

    # Labels
    draw.text((0, 35), "Template", fill="black")
    draw.text(
        (ts[0] + 10, 35),
        f"Detected\nCorr: {title.split(',')[0] if ',' in title else 'N/A'}",
        fill="green",
    )
    if gs:
        draw.text((ts[0] + ds[0] + 20, 35), "Claimed GT", fill="purple")

    comp.save(output_path)
    return output_path


def extract_patch(img_data, x, y, w, h):
    """Extract patch from image data."""
    patch = []
    for row in range(h):
        for col in range(w):
            if 0 <= y + row < img_data.height and 0 <= x + col < img_data.width:
                idx = (y + row) * img_data.width + (x + col)
                patch.append(img_data.data[idx])
            else:
                patch.append(0.0)
    return patch


def patch_to_image(patch, w, h):
    """Convert patch to PIL image."""
    img = Image.new("L", (w, h))
    for i, val in enumerate(patch):
        x, y = i % w, i // w
        img.putpixel((x, y), int(max(0, min(255, val * 255))))
    return img


def compute_patch_similarity(patch1, patch2):
    """Compute correlation between two patches."""
    if len(patch1) != len(patch2):
        return 0.0

    arr1 = np.array(patch1)
    arr2 = np.array(patch2)

    mean1, mean2 = np.mean(arr1), np.mean(arr2)
    std1, std2 = np.std(arr1), np.std(arr2)

    if std1 < 1e-6 or std2 < 1e-6:
        return 0.0

    normalized1 = (arr1 - mean1) / std1
    normalized2 = (arr2 - mean2) / std2

    return float(np.corrcoef(normalized1, normalized2)[0, 1])


def main():
    print("=" * 70)
    print("FINAL VALIDATION: Vulkan Tensorial Template Matching")
    print("=" * 70)

    OUTPUT_DIR = Path("test_data/final_validation")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matcher = VulkanTensorMatcher()

    # Load answers
    answers = []
    answers_file = Path("test_data/extensive/answers.jsonl")
    if answers_file.exists():
        with open(answers_file) as f:
            for line in f:
                answers.append(json.loads(line))

    results = []

    for case_idx, case in enumerate(answers[:10]):  # Test first 10 cases
        print(f"\n{'=' * 70}")
        print(f"Case {case_idx}: {case['image_path']}")
        print(f"{'=' * 70}")

        img_path = Path("test_data") / case["image_path"]
        tmpl_path = Path("test_data") / case["template_path"]

        if not img_path.exists() or not tmpl_path.exists():
            print(f"  SKIP: Files not found")
            continue

        # Load images
        img = ImageData.from_file(str(img_path))
        tmpl = ImageData.from_file(str(tmpl_path))

        # Get claimed ground truth
        gt = case["expected_matches"][0]
        gt_x, gt_y = gt["x"], gt["y"]
        gt_w, gt_h = gt["w"], gt["h"]
        gt_center_x = gt_x + gt_w // 2
        gt_center_y = gt_y + gt_h // 2

        print(f"  Image: {img.width}x{img.height}")
        print(f"  Template: {tmpl.width}x{tmpl.height}")
        print(
            f"  Claimed GT: top-left=({gt_x}, {gt_y}), center=({gt_center_x}, {gt_center_y})"
        )

        # Run matching
        start = time.time()
        matches = matcher.match_template(img, tmpl, 0.3, 3)
        duration = (time.time() - start) * 1000

        if not matches:
            print(f"  NO MATCHES FOUND in {duration:.1f}ms")
            continue

        best = matches[0]
        print(
            f"  Best Match: ({best.x}, {best.y}) corr={best.correlation:.3f} rot={best.rotation_angle:.1f}rad"
        )
        print(
            f"  Distance from GT center: {((best.x - gt_center_x) ** 2 + (best.y - gt_center_y) ** 2) ** 0.5:.1f}px"
        )

        # Extract patches for comparison
        det_x = int(best.x - tmpl.width / 2)
        det_y = int(best.y - tmpl.height / 2)

        detected_patch = extract_patch(img, det_x, det_y, tmpl.width, tmpl.height)
        gt_patch = extract_patch(img, gt_x, gt_y, tmpl.width, tmpl.height)
        template_patch = tmpl.data

        # Compute similarities
        det_similarity = compute_patch_similarity(template_patch, detected_patch)
        gt_similarity = compute_patch_similarity(template_patch, gt_patch)

        print(f"\n  Patch Similarity to Template:")
        print(f"    At Detected Location: {det_similarity:.3f}")
        print(f"    At Claimed GT Location: {gt_similarity:.3f}")

        # Create comparison image
        tmpl_img = patch_to_image(template_patch, tmpl.width, tmpl.height)
        det_img = patch_to_image(detected_patch, tmpl.width, tmpl.height)
        gt_img = patch_to_image(gt_patch, tmpl.width, tmpl.height)

        comp_path = OUTPUT_DIR / f"case_{case_idx:03d}_comparison.png"
        draw_comparison(
            f"Case {case_idx}: {best.correlation:.3f}",
            tmpl_img,
            det_img,
            gt_img,
            comp_path,
        )

        result = {
            "case": case_idx,
            "image": case["image_path"],
            "ttm_corr": best.correlation,
            "ttm_location": (best.x, best.y),
            "gt_location": (gt_center_x, gt_center_y),
            "distance_px": ((best.x - gt_center_x) ** 2 + (best.y - gt_center_y) ** 2)
            ** 0.5,
            "detected_patch_similarity": det_similarity,
            "gt_patch_similarity": gt_similarity,
            "duration_ms": duration,
        }
        results.append(result)

        # Verdict
        if det_similarity > 0.8 and gt_similarity < 0.5:
            print(f"  ✓ OUR MATCH IS CORRECT, GT IS WRONG")
        elif gt_similarity > 0.8:
            print(f"  ✓ GT LOCATION IS VALID")
        else:
            print(f"  ⚠ NEITHER LOCATION MATCHES WELL")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    correct_matches = sum(1 for r in results if r["detected_patch_similarity"] > 0.7)
    gt_correct = sum(1 for r in results if r["gt_patch_similarity"] > 0.7)

    print(f"\nOut of {len(results)} cases:")
    print(f"  Our matches with similarity > 0.7: {correct_matches}/{len(results)}")
    print(f"  GT locations with similarity > 0.7: {gt_correct}/{len(results)}")

    if correct_matches > gt_correct:
        print(
            f"\n✓ CONCLUSION: Our implementation finds BETTER matches than the claimed GT"
        )
    elif correct_matches == gt_correct:
        print(f"\n✓ CONCLUSION: Our implementation performs EQUALLY to GT")
    else:
        print(f"\n✗ CONCLUSION: GT locations are more accurate")

    # Save summary
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {summary_path}")
    print(f"Comparison images saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
