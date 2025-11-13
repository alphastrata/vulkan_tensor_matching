#!/usr/bin/env python3
import json
from pathlib import Path
from PIL import Image
import numpy as np

TEST_DATA = Path("test_data")
TEMPLATES_DIR = TEST_DATA / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

# Use LARGEST distinctive templates
test_cases = [
    # Lenna - large template from distinctive area
    {"image": "lenna.png", "x": 100, "y": 100, "w": 128, "h": 128},
]

answers = []
for i, tc in enumerate(test_cases):
    img_path = TEST_DATA / tc["image"]
    if not img_path.exists():
        continue
    
    img = Image.open(img_path).convert("L")
    img_np = np.array(img)
    
    x, y, w, h = tc["x"], tc["y"], tc["w"], tc["h"]
    tmpl = img_np[y:y+h, x:x+w]
    
    tmpl_path = TEMPLATES_DIR / f"test_{i:03d}.png"
    Image.fromarray(tmpl).save(tmpl_path)
    
    answer = {
        "image_path": tc["image"],
        "template_path": str(tmpl_path.relative_to(TEST_DATA)),
        "expected_matches": [{"x": x, "y": y, "w": w, "h": h, "angle": 0.0}]
    }
    answers.append(answer)
    print(f"Created: {tmpl_path.name} ({w}x{h}) from ({x},{y})")

with open(TEST_DATA / "test_answers.jsonl", "w") as f:
    for a in answers:
        f.write(json.dumps(a) + "\n")

print(f"\nCreated {len(answers)} test cases")
