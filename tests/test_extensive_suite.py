#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from vulkan_tensor_matching import (
    ImageData,
    VulkanNCCMatcher,
    VulkanTensorMatcher,
)

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
EXTENSIVE_DIR = TEST_DATA_DIR / "extensive"
ANSWERS_FILE = EXTENSIVE_DIR / "answers.jsonl"


def load_test_cases():
    if not ANSWERS_FILE.exists():
        return []

    test_cases = []
    with open(ANSWERS_FILE) as f:
        for line in f:
            test_cases.append(json.loads(line))
    return test_cases


class TestExtensiveVulkan:
    @pytest.fixture(scope="class")
    def ncc_matcher(self):
        try:
            return VulkanNCCMatcher()
        except Exception as e:
            pytest.skip(f"Vulkan NCC unavailable: {e}")

    @pytest.fixture(scope="class")
    def tensor_matcher(self):
        try:
            return VulkanTensorMatcher()
        except Exception as e:
            pytest.skip(f"Vulkan Tensor unavailable: {e}")

    def test_synthetic_rotation(self, tensor_matcher):
        """Verify rotation-invariant matching with a simple synthetic case."""
        w, h = 200, 200
        data = np.zeros((h, w), dtype=np.float32)

        # pattern
        pattern = np.zeros((40, 40), dtype=np.float32)
        pattern[10:30, 10:15] = 1.0
        pattern[25:30, 15:30] = 1.0

        # source
        data[80:120, 80:120] = pattern
        img = ImageData(data.flatten().tolist(), w, h, 1)

        # template (rotated 45)
        pat_img = Image.fromarray((pattern * 255).astype(np.uint8))
        rotated_pat = pat_img.rotate(45, resample=Image.BICUBIC, expand=True)
        rotated_data = np.array(rotated_pat).astype(np.float32) / 255.0
        tmpl = ImageData(
            rotated_data.flatten().tolist(), rotated_pat.width, rotated_pat.height, 1
        )

        matches = tensor_matcher.match_template(img, tmpl, 0.0, 10)

        print("\nSynthetic rotation (45°) RAW top 10:")
        for i, m in enumerate(matches):
            print(
                f"  #{i}: ({m.x}, {m.y}) corr={m.correlation:.4f} angle={m.rotation_angle:.1f}°"
            )

        assert len(matches) > 0, (
            "No matches found for synthetic rotation even with 0.0 threshold"
        )
        # Centre in img was (100, 100)
        m = matches[0]
        dx = abs(m.x - 100)
        dy = abs(m.y - 100)
        assert dx <= 5 and dy <= 5, f"Position mismatch: ({m.x}, {m.y})"

    @pytest.mark.parametrize("test_case", load_test_cases())
    def test_vulkan_match(self, ncc_matcher, tensor_matcher, test_case):
        img_path = TEST_DATA_DIR / test_case["image_path"]
        tmpl_path = TEST_DATA_DIR / test_case["template_path"]
        expected = test_case["expected_matches"][0]
        is_rotated = expected.get("angle", 0) != 0

        assert img_path.exists(), f"Image not found: {img_path}"
        assert tmpl_path.exists(), f"Template not found: {tmpl_path}"

        img_data = ImageData.from_file(str(img_path))
        tmpl_data = ImageData.from_file(str(tmpl_path))

        if is_rotated:
            matcher = tensor_matcher
            matcher_name = "Tensor (Rotation-Invariant)"
            threshold = 0.1
        else:
            matcher = ncc_matcher
            matcher_name = "NCC (Standard)"
            threshold = 0.5

        matches = matcher.match_template(
            img_data, tmpl_data, correlation_threshold=threshold, max_matches=10
        )

        found_correct = False
        match_details = []
        for i, m in enumerate(matches):
            dx = abs(m.x - expected["x"])
            dy = abs(m.y - expected["y"])
            pos_ok = dx <= 5 and dy <= 5

            rot_ok = True
            if is_rotated:
                expected_angle = expected["angle"] % 360
                detected_angle = m.rotation_angle % 360
                diff = abs(expected_angle - detected_angle)
                if diff > 180:
                    diff = 360 - diff
                rot_ok = diff <= 10.0  # Relaxed to 10 for real images
                match_details.append(
                    f"#{i}: pos=({m.x}, {m.y}) corr={m.correlation:.4f} "
                    f"angle={m.rotation_angle:.1f}° dist=({dx}, {dy}) "
                    f"rot_diff={diff:.1f}°"
                )
            else:
                match_details.append(
                    f"#{i}: pos=({m.x}, {m.y}) corr={m.correlation:.4f} dist=({dx}, {dy})"
                )

            if pos_ok and rot_ok:
                found_correct = True
                break

        info = "\n".join(match_details)
        expected_info = f"({expected['x']}, {expected['y']})"
        if is_rotated:
            expected_info += f" @ {expected['angle']:.1f}°"

        assert found_correct, (
            f"Correct match not found for {img_path.name} using {matcher_name}.\n"
            f"Expected {expected_info}. Found:\n{info}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
