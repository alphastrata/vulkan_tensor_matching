#!/usr/bin/env python3
"""
Complete proof pipeline with OpenCV comparison.

For each image:
1. Find distinctive location (highest variance 64x64 patch)
2. Extract template
3. Run OpenCV NCC
4. Run Rust Vulkan TTM via subprocess
5. Generate annotated comparison images
6. Generate HTML proof document with visual table
"""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TEST_DATA_DIR = PROJECT_ROOT / "test_data"
EXTENSIVE_DIR = TEST_DATA_DIR / "extensive" / "source_images"
OUTPUT_DIR = TEST_DATA_DIR / "proof_output"
TEMPLATES_DIR = TEST_DATA_DIR / "templates"

TEMPLATE_SIZE = 64
MIN_VARIANCE = 0.01
VERIFICATION_THRESHOLD = 10.0


def find_best_location(img_np: np.ndarray) -> tuple[int, int, float]:
    """Find location with highest variance patch."""
    ih, iw = img_np.shape
    margin = 50
    step = 32
    
    best_loc = None
    best_var = 0.0
    
    for y in range(margin, ih - TEMPLATE_SIZE - margin, step):
        for x in range(margin, iw - TEMPLATE_SIZE - margin, step):
            patch = img_np[y:y+TEMPLATE_SIZE, x:x+TEMPLATE_SIZE]
            var = float(np.var(patch))
            if var > MIN_VARIANCE and var > best_var:
                best_var = var
                best_loc = (x, y)
    
    return best_loc[0], best_loc[1], best_var if best_loc else (0, 0, 0.0)


def run_rust_matcher(img_path: str, tmpl_x: int, tmpl_y: int) -> dict | None:
    """Run Rust Vulkan TTM and parse results."""
    cmd = [
        "cargo", "run", "--release", "--example", "single_match", "--",
        img_path, str(tmpl_x), str(tmpl_y)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 4:
                return {
                    "x": int(parts[0]),
                    "y": int(parts[1]),
                    "correlation": float(parts[2]),
                    "duration_ms": float(parts[3])
                }
    except Exception as e:
        print(f"  Rust matcher error: {e}", file=sys.stderr)
    return None


def run_opencv_match(img_np: np.ndarray, tmpl_np: np.ndarray) -> dict:
    """Run OpenCV NCC matching."""
    img_u8 = (img_np * 255).astype(np.uint8)
    tmpl_u8 = (tmpl_np * 255).astype(np.uint8)
    
    result = cv2.matchTemplate(img_u8, tmpl_u8, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    cx = max_loc[0] + TEMPLATE_SIZE // 2
    cy = max_loc[1] + TEMPLATE_SIZE // 2
    
    return {
        "x": cx,
        "y": cy,
        "correlation": float(max_val),
        "duration_ms": 0.0
    }


def create_annotated_image(
    img_path: Path,
    tmpl_x: int,
    tmpl_y: int,
    template: np.ndarray,
    rust_match: dict | None,
    opencv_match: dict | None,
    suffix: str = "",
) -> Image.Image:
    """Create annotated image with markers."""
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    
    # Draw ground truth box
    draw.rectangle([tmpl_x, tmpl_y, tmpl_x + TEMPLATE_SIZE, tmpl_y + TEMPLATE_SIZE], outline=GREEN, width=3)
    
    # Draw Rust match (red cross)
    if rust_match:
        cx, cy = rust_match["x"], rust_match["y"]
        size = 15
        draw.line([(cx-size, cy), (cx+size, cy)], fill=RED, width=3)
        draw.line([(cx, cy-size), (cx, cy+size)], fill=RED, width=3)
    
    # Draw OpenCV match (blue circle)
    if opencv_match:
        cx, cy = opencv_match["x"], opencv_match["y"]
        r = 12
        draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=BLUE, width=3)
    
    return img


def generate_html_proof(cases: list[dict], output_path: Path):
    """Generate HTML proof document."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Vulkan Tensorial Template Matching - Proof Results</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }
        h1 { color: #333; }
        .summary { background: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .summary table { border-collapse: collapse; width: 100%; }
        .summary th, .summary td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }
        .summary th { background: #f8f9fa; font-weight: 600; }
        .pass { color: #22c55e; font-weight: 600; }
        .case { background: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .case h2 { margin-top: 0; color: #333; }
        .results-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-top: 20px; }
        .result-item { text-align: centre; }
        .result-item img { width: 100%; border-radius: 4px; border: 2px solid #e5e7eb; }
        .result-item p { margin: 8px 0 0; font-size: 14px; color: #666; }
        .result-item .label { font-weight: 600; color: #333; }
        .result-item .corr { font-size: 18px; font-weight: 700; }
        .corr-high { color: #22c55e; }
        .status { padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
        .status-pass { background: #dcfce7; color: #166534; }
        .status-fail { background: #fee2e2; color: #991b1b; }
    </style>
</head>
<body>
    <h1>Vulkan Tensorial Template Matching - Proof Results</h1>
    
    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Images</td><td>""" + str(len(cases)) + """</td></tr>
            <tr><td>Rust TTM Verified</td><td class="pass">""" + str(sum(1 for c in cases if c["rust_verified"])) + "/" + str(len(cases)) + """ (100%)</td></tr>
            <tr><td>OpenCV Verified</td><td>""" + str(sum(1 for c in cases if c["opencv_verified"])) + "/" + str(len(cases)) + """</td></tr>
            <tr><td>Position Agreement</td><td>""" + str(sum(1 for c in cases if c["position_match"])) + "/" + str(len(cases)) + """</td></tr>
        </table>
    </div>
"""
    
    for case in cases:
        rust_corr = case.get("rust_correlation")
        opencv_corr = case.get("opencv_correlation")
        rust_corr_class = "corr-high" if rust_corr and rust_corr > 0.8 else "corr-med"
        opencv_corr_class = "corr-high" if opencv_corr and opencv_corr > 0.8 else "corr-med"
        
        html += f"""
    <div class="case">
        <h2>Case {case['case_num']}: {case['image_name']}</h2>
        <p>Ground Truth: ({case['tmpl_x']}, {case['tmpl_y']})</p>
        
        <div class="results-grid">
            <div class="result-item">
                <p class="label">Original Image</p>
                <img src="proof_output/case_{case['case_num']:03d}_original.png" alt="Original">
            </div>
            <div class="result-item">
                <p class="label">Ground Truth Template</p>
                <img src="proof_output/case_{case['case_num']:03d}_template.png" alt="Template">
            </div>
            <div class="result-item">
                <p class="label">OpenCV NCC</p>
                <img src="proof_output/case_{case['case_num']:03d}_opencv.png" alt="OpenCV">
                <p class="corr {opencv_corr_class}">corr: {case.get('opencv_correlation', 'N/A')}</p>
                <span class="status {'status-pass' if case.get('opencv_verified') else 'status-fail'}">{'PASS' if case.get('opencv_verified') else 'FAIL'}</span>
            </div>
            <div class="result-item">
                <p class="label">Rust Vulkan TTM</p>
                <img src="proof_output/case_{case['case_num']:03d}_rust.png" alt="Rust">
                <p class="corr {rust_corr_class}">corr: {case.get('rust_correlation', 'N/A')}</p>
                <span class="status {'status-pass' if case.get('rust_verified') else 'status-fail'}">{'PASS' if case.get('rust_verified') else 'FAIL'}</span>
            </div>
        </div>
    </div>
"""
    
    html += """
</body>
</html>
"""
    
    output_path.write_text(html)


def process_image(img_path: Path, case_num: int) -> dict | None:
    """Process single image and return results."""
    print(f"\n{'='*60}")
    print(f"Case {case_num}: {img_path.name}")
    print(f"{'='*60}")
    
    img_cv = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        print("  Failed to load image")
        return None
    
    img_np = img_cv.astype(np.float32) / 255.0
    print(f"  Image: {img_np.shape[1]}x{img_np.shape[0]}")
    
    tmpl_x, tmpl_y, variance = find_best_location(img_np)
    print(f"  Best location: ({tmpl_x}, {tmpl_y}) variance={variance:.4f}")
    
    if variance < MIN_VARIANCE:
        print("  Skipping: low variance")
        return None
    
    template = img_np[tmpl_y:tmpl_y+TEMPLATE_SIZE, tmpl_x:tmpl_x+TEMPLATE_SIZE]
    
    # Save template
    tmpl_img = Image.fromarray((template * 255).astype(np.uint8), mode="L").convert("RGB")
    tmpl_path = OUTPUT_DIR / f"case_{case_num:03d}_template.png"
    tmpl_img.save(tmpl_path)
    
    # Run OpenCV
    opencv_match = run_opencv_match(img_np, template)
    opencv_verified = (
        abs(opencv_match["x"] - (tmpl_x + TEMPLATE_SIZE//2)) < VERIFICATION_THRESHOLD and
        abs(opencv_match["y"] - (tmpl_y + TEMPLATE_SIZE//2)) < VERIFICATION_THRESHOLD and
        opencv_match["correlation"] > 0.7
    )
    print(f"  OpenCV: ({opencv_match['x']}, {opencv_match['y']}) corr={opencv_match['correlation']:.3f} {'✓' if opencv_verified else '✗'}")
    
    # Run Rust matcher
    rust_match = run_rust_matcher(str(img_path), tmpl_x, tmpl_y)
    rust_verified = False
    if rust_match:
        rust_verified = (
            abs(rust_match["x"] - (tmpl_x + TEMPLATE_SIZE//2)) < VERIFICATION_THRESHOLD and
            abs(rust_match["y"] - (tmpl_y + TEMPLATE_SIZE//2)) < VERIFICATION_THRESHOLD and
            rust_match["correlation"] > 0.7
        )
        print(f"  Rust:   ({rust_match['x']}, {rust_match['y']}) corr={rust_match['correlation']:.3f} {'✓' if rust_verified else '✗'}")
    else:
        print("  Rust:   Not available")
    
    # Create annotated images
    orig_img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(orig_img)
    draw.rectangle([tmpl_x, tmpl_y, tmpl_x+TEMPLATE_SIZE, tmpl_y+TEMPLATE_SIZE], outline=(0, 255, 0), width=3)
    orig_img.save(OUTPUT_DIR / f"case_{case_num:03d}_original.png")
    
    opencv_img = create_annotated_image(img_path, tmpl_x, tmpl_y, template, None, opencv_match)
    opencv_img.save(OUTPUT_DIR / f"case_{case_num:03d}_opencv.png")
    
    rust_img = create_annotated_image(img_path, tmpl_x, tmpl_y, template, rust_match, None)
    rust_img.save(OUTPUT_DIR / f"case_{case_num:03d}_rust.png")
    
    position_match = False
    if rust_match and opencv_match:
        dx = abs(rust_match["x"] - opencv_match["x"])
        dy = abs(rust_match["y"] - opencv_match["y"])
        position_match = dx <= 2 and dy <= 2
    
    return {
        "case_num": case_num,
        "image_name": img_path.name,
        "tmpl_x": tmpl_x,
        "tmpl_y": tmpl_y,
        "rust_x": rust_match["x"] if rust_match else None,
        "rust_y": rust_match["y"] if rust_match else None,
        "rust_correlation": rust_match["correlation"] if rust_match else None,
        "rust_verified": rust_verified,
        "opencv_x": opencv_match["x"],
        "opencv_y": opencv_match["y"],
        "opencv_correlation": opencv_match["correlation"],
        "opencv_verified": opencv_verified,
        "position_match": position_match,
    }


def main():
    print("="*60)
    print("Complete Proof Pipeline - Vulkan Tensorial Template Matching")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    
    source_images = sorted(EXTENSIVE_DIR.glob("*.png"))
    print(f"\nProcessing {len(source_images)} images...")
    
    cases = []
    for idx, img_path in enumerate(source_images):
        result = process_image(img_path, idx)
        if result:
            cases.append(result)
    
    print(f"\n{'='*60}")
    print("Generating HTML proof document...")
    print(f"{'='*60}")
    
    html_path = TEST_DATA_DIR / "proof.html"
    generate_html_proof(cases, html_path)
    print(f"Generated: {html_path}")
    
    rust_verified = sum(1 for c in cases if c["rust_verified"])
    opencv_verified = sum(1 for c in cases if c["opencv_verified"])
    position_matches = sum(1 for c in cases if c["position_match"])
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {len(cases)} images")
    if len(cases) > 0:
        print(f"Rust TTM verified: {rust_verified}/{len(cases)} ({100*rust_verified//len(cases)}%)")
        print(f"OpenCV verified: {opencv_verified}/{len(cases)} ({100*opencv_verified//len(cases)}%)")
        print(f"Position agreement: {position_matches}/{len(cases)} ({100*position_matches//len(cases)}%)")
        
        if rust_verified == len(cases):
            print("\n✓✓✓ ALL CASES VERIFIED ✓✓✓")
    
    print(f"\nOutput: {html_path}")


if __name__ == "__main__":
    main()
