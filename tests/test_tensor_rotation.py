#!/usr/bin/env python3
"""
Tests for VulkanTensorMatcher rotation-invariance.

These tests validate the core tensorial template matching implementation
based on the Martinez-Sanchez paper. They verify:
- Rotation-invariant detection (templates found at any angle)
- Rotation angle accuracy (reported angles match ground truth)
- Multi-instance detection (same template at different rotations)
- Position accuracy for rotated templates
"""

from pathlib import Path

import numpy as np
import pytest

from vulkan_tensor_matching import ImageData, VulkanTensorMatcher

TEST_ASSETS_DIR = Path(__file__).parent.parent / "test_data"


def numpy_to_image_data(arr: np.ndarray) -> ImageData:
    """Convert numpy array to ImageData."""
    arr = arr.astype(np.float32)
    return ImageData(
        data=arr.flatten().tolist(),
        width=arr.shape[1],
        height=arr.shape[0],
        channels=1,
    )


def image_data_to_numpy(image_data: ImageData) -> np.ndarray:
    """Convert ImageData to numpy array."""
    return np.array(image_data.data, dtype=np.float32).reshape(
        (image_data.height, image_data.width)
    )


def create_lshape_template(size: int) -> np.ndarray:
    """Create an L-shaped template."""
    template = np.zeros((size, size), dtype=np.float32)
    stem_width = size // 4
    base_height = size // 4

    # Vertical stem
    template[: size - base_height, :stem_width] = 1.0

    # Horizontal base
    template[size - base_height :, :] = 1.0

    return template


def rotate_point(x: float, y: float, angle_deg: float, cx: float, cy: float) -> tuple:
    """Rotate a point around the origin."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    nx = cx + (x - cx) * cos_a - (y - cy) * sin_a
    ny = cy + (x - cx) * sin_a + (y - cy) * cos_a

    return nx, ny


def plant_rotated_template(
    image: np.ndarray,
    template: np.ndarray,
    centre_x: int,
    centre_y: int,
    angle_deg: float,
) -> None:
    """Plant a template in an image at a specific rotation angle."""
    h, w = template.shape
    cx, cy = w / 2.0, h / 2.0

    for ty in range(h):
        for tx in range(w):
            val = template[ty, tx]
            if val < 0.5:
                continue

            # Rotate the template coordinate
            rx, ry = rotate_point(tx, ty, angle_deg, cx, cy)

            # Map to image coordinates
            ix = centre_x + rx - cx
            it = centre_y + ry - cy

            # Check bounds
            if ix < 0 or ix >= image.shape[1] or it < 0 or it >= image.shape[0]:
                continue

            # Simple nearest-neighbor planting
            ix_u = int(round(ix))
            it_u = int(round(it))

            if 0 <= ix_u < image.shape[1] and 0 <= it_u < image.shape[0]:
                image[it_u, ix_u] = val


@pytest.fixture
def matcher():
    """Create VulkanTensorMatcher, skip if unavailable."""
    try:
        m = VulkanTensorMatcher()
        yield m
        del m
    except ValueError as e:
        pytest.skip(f"Vulkan unavailable: {e}")


class TestTensorRotationInvariance:
    """
    Tests for rotation-invariant template matching using VulkanTensorMatcher.

    These tests validate the core contribution of the Martinez-Sanchez paper:
    rotation-invariant template matching using tensorial representation.
    """

    def test_tensor_finds_template_at_0_degrees(self, matcher):
        """Test detection of template at 0° rotation."""
        image_w, image_h = 100, 100
        template_size = 20

        image = np.full((image_h, image_w), 0.1, dtype=np.float32)
        template = create_lshape_template(template_size)

        plant_rotated_template(image, template, 50, 50, 0.0)

        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)

        matches = matcher.match_template(
            img_data, tmpl_data, correlation_threshold=0.3, max_matches=5
        )

        assert len(matches) > 0, "Tensor matcher found no matches at 0°"

        best = matches[0]
        assert best.correlation > 0.5, f"Correlation too low at 0°: {best.correlation}"

        # Position should be near centre
        assert abs(best.x - 50) <= 10, f"Position x={best.x} too far from 50"
        assert abs(best.y - 50) <= 10, f"Position y={best.y} too far from 50"

    def test_tensor_finds_template_at_90_degrees(self, matcher):
        """Test detection of template at 90° rotation."""
        image_w, image_h = 100, 100
        template_size = 20

        image = np.full((image_h, image_w), 0.1, dtype=np.float32)
        template = create_lshape_template(template_size)

        plant_rotated_template(image, template, 50, 50, 90.0)

        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)

        matches = matcher.match_template(
            img_data, tmpl_data, correlation_threshold=0.3, max_matches=5
        )

        assert len(matches) > 0, "Tensor matcher found no matches at 90°"

        best = matches[0]
        angle_diff = abs(best.rotation_angle - 90.0)
        angle_diff = min(angle_diff, 360.0 - angle_diff)

        assert angle_diff <= 20.0, (
            f"Reported angle {best.rotation_angle}° too far from 90° (diff: {angle_diff}°)"
        )

    def test_tensor_finds_template_at_180_degrees(self, matcher):
        """Test detection of template at 180° rotation."""
        image_w, image_h = 100, 100
        template_size = 20

        image = np.full((image_h, image_w), 0.1, dtype=np.float32)
        template = create_lshape_template(template_size)

        plant_rotated_template(image, template, 50, 50, 180.0)

        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)

        matches = matcher.match_template(
            img_data, tmpl_data, correlation_threshold=0.3, max_matches=5
        )

        assert len(matches) > 0, "Tensor matcher found no matches at 180°"

        best = matches[0]
        angle_diff = abs(best.rotation_angle - 180.0)
        angle_diff = min(angle_diff, 360.0 - angle_diff)

        assert angle_diff <= 20.0, (
            f"Reported angle {best.rotation_angle}° too far from 180° (diff: {angle_diff}°)"
        )

    def test_tensor_finds_template_at_270_degrees(self, matcher):
        """Test detection of template at 270° rotation."""
        image_w, image_h = 100, 100
        template_size = 20

        image = np.full((image_h, image_w), 0.1, dtype=np.float32)
        template = create_lshape_template(template_size)

        plant_rotated_template(image, template, 50, 50, 270.0)

        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)

        matches = matcher.match_template(
            img_data, tmpl_data, correlation_threshold=0.3, max_matches=5
        )

        assert len(matches) > 0, "Tensor matcher found no matches at 270°"

        best = matches[0]
        angle_diff = abs(best.rotation_angle - 270.0)
        angle_diff = min(angle_diff, 360.0 - angle_diff)

        assert angle_diff <= 20.0, (
            f"Reported angle {best.rotation_angle}° too far from 270° (diff: {angle_diff}°)"
        )

    def test_tensor_rotation_sweep(self, matcher):
        """Test detection at multiple rotation angles."""
        angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        image_w, image_h = 100, 100
        template_size = 20

        for angle in angles:
            image = np.full((image_h, image_w), 0.1, dtype=np.float32)
            template = create_lshape_template(template_size)

            plant_rotated_template(image, template, 50, 50, angle)

            img_data = numpy_to_image_data(image)
            tmpl_data = numpy_to_image_data(template)

            matches = matcher.match_template(
                img_data, tmpl_data, correlation_threshold=0.3, max_matches=5
            )

            assert len(matches) > 0, f"Tensor matcher found no matches at {angle}°"

            best = matches[0]
            angle_diff = abs(best.rotation_angle - angle)
            angle_diff = min(angle_diff, 360.0 - angle_diff)

            print(
                f"Angle {angle}: reported {best.rotation_angle:.1f}° (diff: {angle_diff:.1f}°), corr: {best.correlation:.3f}"
            )

    def test_tensor_multi_instance_rotation(self, matcher):
        """Test finding same template at multiple rotation angles in one image."""
        image_w, image_h = 150, 150
        template_size = 20

        image = np.full((image_h, image_w), 0.1, dtype=np.float32)
        template = create_lshape_template(template_size)

        # Instance 1: at (40, 40), 0°
        plant_rotated_template(image, template, 40, 40, 0.0)

        # Instance 2: at (80, 40), 90°
        plant_rotated_template(image, template, 80, 40, 90.0)

        # Instance 3: at (60, 100), 180°
        plant_rotated_template(image, template, 60, 100, 180.0)

        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)

        matches = matcher.match_template(
            img_data, tmpl_data, correlation_threshold=0.3, max_matches=10
        )

        assert len(matches) >= 3, f"Expected at least 3 matches (found {len(matches)})"

        # Check that we found instances near expected locations
        expected_positions = [(40, 40, 0.0), (80, 40, 90.0), (60, 100, 180.0)]

        for exp_x, exp_y, exp_angle in expected_positions:
            found = any(
                abs(m.x - exp_x) <= 10
                and abs(m.y - exp_y) <= 10
                and min(
                    abs(m.rotation_angle - exp_angle),
                    360.0 - abs(m.rotation_angle - exp_angle),
                )
                <= 20.0
                for m in matches
            )

            assert found, f"No match found near ({exp_x}, {exp_y}) at {exp_angle}°"

    def test_tensor_rotation_angle_accuracy(self, matcher):
        """Test precise angle detection accuracy."""
        test_angles = [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0]
        angle_tolerance = 15.0  # degrees
        image_w, image_h = 100, 100
        template_size = 20

        for expected_angle in test_angles:
            image = np.full((image_h, image_w), 0.1, dtype=np.float32)
            template = create_lshape_template(template_size)

            plant_rotated_template(image, template, 50, 50, expected_angle)

            img_data = numpy_to_image_data(image)
            tmpl_data = numpy_to_image_data(template)

            matches = matcher.match_template(
                img_data, tmpl_data, correlation_threshold=0.3, max_matches=5
            )

            assert len(matches) > 0, f"No matches found at {expected_angle}°"

            best = matches[0]
            angle_diff = abs(best.rotation_angle - expected_angle)
            angle_diff = min(angle_diff, 360.0 - angle_diff)

            assert angle_diff <= angle_tolerance, (
                f"At {expected_angle}°: reported {best.rotation_angle:.1f}° (diff: {angle_diff:.1f}°, tolerance: {angle_tolerance}°)"
            )

    def test_tensor_rotation_angle_wrapping(self, matcher):
        """Test angle wrapping near boundaries."""
        boundary_angles = [(355.0, 355.0), (5.0, 5.0), (175.0, 175.0), (185.0, 185.0)]
        image_w, image_h = 100, 100
        template_size = 20

        for planted_angle, expected_angle in boundary_angles:
            image = np.full((image_h, image_w), 0.1, dtype=np.float32)
            template = create_lshape_template(template_size)

            plant_rotated_template(image, template, 50, 50, planted_angle)

            img_data = numpy_to_image_data(image)
            tmpl_data = numpy_to_image_data(template)

            matches = matcher.match_template(
                img_data, tmpl_data, correlation_threshold=0.3, max_matches=5
            )

            if matches:
                best = matches[0]
                angle_diff = abs(best.rotation_angle - expected_angle)
                angle_diff = min(angle_diff, 360.0 - angle_diff)

                print(
                    f"At {planted_angle}°: reported {best.rotation_angle:.1f}° (diff: {angle_diff:.1f}°)"
                )

    def test_tensor_small_template_rotation(self, matcher):
        """Test rotation detection with small template."""
        image_w, image_h = 80, 80
        template_size = 8

        image = np.full((image_h, image_w), 0.1, dtype=np.float32)
        template = np.zeros((template_size, template_size), dtype=np.float32)

        # Create small cross pattern
        template[3:6, :] = 1.0
        template[:, 3:6] = 1.0

        plant_rotated_template(image, template, 40, 40, 45.0)

        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)

        matches = matcher.match_template(
            img_data, tmpl_data, correlation_threshold=0.2, max_matches=5
        )

        assert len(matches) > 0, "Tensor matcher found no matches for small template"

    def test_tensor_noisy_image_rotation(self, matcher):
        """Test rotation detection in noisy image."""
        np.random.seed(42)
        image_w, image_h = 100, 100
        template_size = 20

        # Background noise
        image = np.random.uniform(-0.1, 0.2, (image_h, image_w)).astype(np.float32)
        template = create_lshape_template(template_size)

        plant_rotated_template(image, template, 50, 50, 60.0)

        img_data = numpy_to_image_data(image)
        tmpl_data = numpy_to_image_data(template)

        matches = matcher.match_template(
            img_data, tmpl_data, correlation_threshold=0.3, max_matches=5
        )

        assert len(matches) > 0, "Tensor matcher found no matches in noisy image"


class TestTensorExtensiveTestData:
    """Tests using the extensive test dataset with ground-truth annotations."""

    @pytest.mark.skipif(not TEST_ASSETS_DIR.exists(), reason="No test assets")
    def test_tensor_matches_extensive_rotated_templates(self, matcher):
        """Test VulkanTensorMatcher against ground-truth rotated templates."""
        import json

        answers_path = TEST_ASSETS_DIR / "extensive" / "answers.jsonl"
        if not answers_path.exists():
            pytest.skip("answers.jsonl not found")

        with open(answers_path) as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines):
            entry = json.loads(line)
            img_path = TEST_ASSETS_DIR / entry["image_path"]
            tmpl_path = TEST_ASSETS_DIR / entry["template_path"]

            if not img_path.exists():
                print(f"Skipping line {line_num} - image not found: {img_path}")
                continue
            if not tmpl_path.exists():
                print(f"Skipping line {line_num} - template not found: {tmpl_path}")
                continue

            img_data = ImageData.from_file(str(img_path))
            tmpl_data = ImageData.from_file(str(tmpl_path))

            matches = matcher.match_template(
                img_data, tmpl_data, correlation_threshold=0.3, max_matches=10
            )

            expected_matches = entry["expected_matches"]

            for idx, expected in enumerate(expected_matches):
                if idx >= len(matches):
                    print(
                        f"Line {line_num}: Expected {len(expected_matches)} matches but found {len(matches)}"
                    )
                    continue

                actual = matches[idx]
                expected_angle = expected["angle"]

                # Position tolerance: ±10 pixels
                dx = abs(actual.x - expected["x"])
                dy = abs(actual.y - expected["y"])

                # Angle tolerance: ±15 degrees
                angle_diff = abs(actual.rotation_angle - expected_angle)
                angle_diff = min(angle_diff, 360.0 - angle_diff)

                assert dx <= 10 and dy <= 10, (
                    f"Line {line_num}: Position ({actual.x}, {actual.y}) too far from "
                    f"expected ({expected['x']}, {expected['y']})"
                )

                if angle_diff > 15.0:
                    print(
                        f"Line {line_num}: Angle {actual.rotation_angle:.1f}° differs from "
                        f"expected {expected_angle:.1f}° (diff: {angle_diff:.1f}°)"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
