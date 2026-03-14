#!/usr/bin/env python3
"""
Generate proof images showing template matches.
Uses Rust test results to create annotated images.
"""

import subprocess
import json
from pathlib import Path
from PIL import Image, ImageDraw

TEST_DATA_DIR = Path("test_assets")
OUTPUT_DIR = Path("test_data") / "proof_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_match(image_path, matches, gt_matches, output_path):
    """Draw matches on image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    w, h = img.size
    radius = max(5, min(w, h) // 50)

    # Draw ground truth (purple)
    for gt in gt_matches:
        x, y = gt["x"], gt["y"]
        # Purple circle
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            outline=(128, 0, 128),
            width=3,
        )
        # Purple cross
        draw.line([x - 10, y, x + 10, y], fill=(128, 0, 128), width=2)
        draw.line([x, y - 10, x, y + 10], fill=(128, 0, 128), width=2)
        draw.text((x + 15, y - 15), f"GT", fill=(128, 0, 128))

    # Draw detected matches (green, size by correlation)
    for i, m in enumerate(matches[:5]):
        x, y = m["x"], m["y"]
        corr = m["correlation"]

        # Color: green for good matches, yellow for weaker
        r = int(255 * (1 - min(1.0, max(0.0, corr))))
        g = 255
        b = int(100 * (1 - min(1.0, max(0.0, corr))))
        color = (r, g, b)

        # Circle size by correlation
        m_radius = int(radius * (0.5 + 0.5 * min(1.0, max(0.0, corr))))

        draw.ellipse(
            [x - m_radius, y - m_radius, x + m_radius, y + m_radius],
            outline=color,
            width=2,
        )
        draw.text((x + 15, y + 15), f"#{i + 1}: {corr:.2f}", fill=color)

    img.save(output_path)
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("Generating Proof Images")
    print("=" * 60)

    # Run example to get actual matches
    print("\nRunning example to get match data...")
    result = subprocess.run(
        ["cargo", "run", "--release", "--example", "lenna_vulkan_matching"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
    )

    # Parse matches from output
    lenna_matches = []
    for line in result.stdout.split("\n"):
        if "Match" in line and "x=" in line:
            try:
                # Match 1: x=480, y=197, corr=58446.445, rot=0.0
                parts = line.split(":")[1].strip().split(",")
                x = int(parts[0].split("=")[1].strip())
                y = int(parts[1].split("=")[1].strip())
                corr = float(parts[2].split("=")[1].strip())
                lenna_matches.append({"x": x, "y": y, "correlation": corr})
            except:
                pass

    print(f"Found {len(lenna_matches)} Lenna matches")

    # Generate Lenna proof image
    if lenna_matches:
        print("\nGenerating Lenna proof image...")
        img_path = TEST_DATA_DIR / "lenna.png"

        if img_path.exists():
            # Ground truth centre (approx 45, 95)
            gt = [{"x": 45, "y": 95}]

            output = OUTPUT_DIR / "lenna_matches.png"
            draw_match(str(img_path), lenna_matches, gt, output)

            # Print summary
            print(f"\nLenna Results:")
            print(f"  Ground Truth: (45, 95)")
            for i, m in enumerate(lenna_matches[:3]):
                dist = ((m["x"] - 45) ** 2 + (m["y"] - 95) ** 2) ** 0.5
                status = "✓" if dist < 15 else "✗"
                print(
                    f"  Match {i + 1}: ({m['x']}, {m['y']}) corr={m['correlation']:.4f} dist={dist:.1f}px {status}"
                )

    print("\n" + "=" * 60)
    print("Proof images generated in:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
