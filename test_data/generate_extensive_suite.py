#!/usr/bin/env python3
import os
import json
import random
import urllib.request
from pathlib import Path
from PIL import Image
import numpy as np

# Configuration
NUM_IMAGES = 20
IMAGE_SIZE = (800, 600)  # (W, H)
MIN_TEMPLATE_SIZE = 40
MAX_TEMPLATE_SIZE = 100
EXTENSIVE_DIR = Path(__file__).parent / "extensive"
SOURCE_DIR = EXTENSIVE_DIR / "source_images"
TEMPLATE_DIR = EXTENSIVE_DIR / "templates"
ANSWERS_FILE = EXTENSIVE_DIR / "answers.jsonl"


def setup_directories():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)


def download_image(index):
    url = f"https://picsum.photos/{IMAGE_SIZE[0]}/{IMAGE_SIZE[1]}?random={index}"
    try:
        response = urllib.request.urlopen(url)
        data = response.read()
        image_path = SOURCE_DIR / f"source_{index}.png"
        with open(image_path, "wb") as f:
            f.write(data)
        # Convert to grayscale immediately to ensure consistent format
        with Image.open(image_path) as img:
            img.convert("L").save(image_path)
        return image_path
    except Exception as e:
        print(f"Failed to download image {index}: {e}")
        return None


def generate_template(image_path, index):
    try:
        img = Image.open(image_path)
    except Exception:
        return None

    w, h = img.size

    best_var = -1
    best_rect = None

    # Try to find a high-variance region
    img_np = np.array(img)
    for _ in range(50):
        tw = random.randint(MIN_TEMPLATE_SIZE, min(MAX_TEMPLATE_SIZE, w // 4))
        th = random.randint(MIN_TEMPLATE_SIZE, min(MAX_TEMPLATE_SIZE, h // 4))
        tx = random.randint(0, w - tw)
        ty = random.randint(0, h - th)

        region = img_np[ty : ty + th, tx : tx + tw]
        variance = np.var(region)

        if variance > best_var and variance > 200:
            best_var = variance
            best_rect = (tx, ty, tw, th)

    if best_rect is None:
        return None

    tx, ty, tw, th = best_rect
    template = img.crop((tx, ty, tx + tw, ty + th))

    # Random rotation for every second test case
    angle = 0
    if index % 2 == 1:
        angle = random.uniform(10, 350)
        # Use Image.BICUBIC for rotation, expand=True to keep all pixels
        # but we want to simulate how it looks in source, so we use expand=True
        # and then match. Wait, full tensorial matching handles this.
        template = template.rotate(angle, resample=Image.BICUBIC, expand=True)

    template_path = TEMPLATE_DIR / f"template_{index}.png"
    template.save(template_path)

    # Calculate center in original image
    cx = tx + tw // 2
    cy = ty + th // 2

    return {
        "image_path": str(image_path.relative_to(EXTENSIVE_DIR.parent)),
        "template_path": str(template_path.relative_to(EXTENSIVE_DIR.parent)),
        "expected_matches": [
            {
                "x": int(cx),
                "y": int(cy),
                "w": int(tw),
                "h": int(th),
                "angle": float(angle),
            }
        ],
    }


def main():
    setup_directories()
    results = []

    print(f"Generating {NUM_IMAGES} extensive test cases (including rotations)...")
    for i in range(NUM_IMAGES):
        print(f"Processing {i + 1}/{NUM_IMAGES}...")
        img_path = download_image(i)
        if img_path:
            answer = generate_template(img_path, i)
            if answer:
                results.append(answer)

    with open(ANSWERS_FILE, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    print(f"Successfully generated {len(results)} test cases.")
    print(f"Answers saved to {ANSWERS_FILE}")


if __name__ == "__main__":
    main()
