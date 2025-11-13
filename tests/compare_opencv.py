#!/usr/bin/env python3
"""Direct comparison: Our Vulkan vs OpenCV on SAME template."""

import cv2
import numpy as np
from pathlib import Path
import time
import math

from vulkan_tensor_matching import ImageData, VulkanTensorMatcher

TEST_DATA = Path("test_data")

img_path = TEST_DATA / "extensive/source_images/source_0.png"
tmpl_path = TEST_DATA / "extensive/templates/template_0.png"

print("="*60)
print("Direct Comparison: Vulkan vs OpenCV")
print("="*60)

img_cv = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
tmpl_cv = cv2.imread(str(tmpl_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
img_ours = ImageData.from_file(str(img_path))
tmpl_ours = ImageData.from_file(str(tmpl_path))

print(f"Image: {img_cv.shape[1]}x{img_cv.shape[0]}")
print(f"Template: {tmpl_cv.shape[1]}x{tmpl_cv.shape[0]}")

gt_x, gt_y = 527, 283
gt_cx = gt_x + tmpl_cv.shape[1]//2
gt_cy = gt_y + tmpl_cv.shape[0]//2
print(f"\nGround Truth: ({gt_x}, {gt_y}) center=({gt_cx}, {gt_cy})")

# OpenCV
print("\nOpenCV TM_CCOEFF_NORMED...")
start = time.time()
result = cv2.matchTemplate(img_cv, tmpl_cv, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
opencv_time = time.time() - start

opencv_cx = max_loc[0] + tmpl_cv.shape[1]//2
opencv_cy = max_loc[1] + tmpl_cv.shape[0]//2
opencv_dist = math.sqrt((opencv_cx - gt_cx)**2 + **(opencv_cy - gt_cy)2)

print(f"  Location: {max_loc} center=({opencv_cx}, {opencv_cy})")
print(f"  Correlation: {max_val:.4f}")
print(f"  Distance from GT: {opencv_dist:.1f}px")
print(f"  Time: {opencv_time:.3f}s")

# Vulkan
print("\nVulkan TTM...")
start = time.time()
matches = VulkanTensorMatcher().match_template(img_ours, tmpl_ours, 0.3, 1)
vulkan_time = time.time() - start

if matches:
    m = matches[0]
    vulkan_dist = math.sqrt((m.x - gt_cx)**2 + **(m.y - gt_cy)2)
    print(f"  Location: ({m.x}, {m.y})")
    print(f"  Correlation: {m.correlation:.4f}")
    print(f"  Distance from GT: {vulkan_dist:.1f}px")
    print(f"  Time: {vulkan_time:.3f}s")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# Check GT patch
gt_patch = img_cv[gt_y:gt_y+tmpl_cv.shape[0], gt_x:gt_x+tmpl_cv.shape[1]]
gt_corr = np.corrcoef(gt_patch.flatten(), tmpl_cv.flatten())[0,1]
print(f"Template vs GT patch correlation: {gt_corr:.4f}")

# Check OpenCV patch
opencv_patch = img_cv[max_loc[1]:max_loc[1]+tmpl_cv.shape[0], max_loc[0]:max_loc[0]+tmpl_cv.shape[1]]
opencv_patch_corr = np.corrcoef(opencv_patch.flatten(), tmpl_cv.flatten())[0,1]
print(f"Template vs OpenCV patch correlation: {opencv_patch_corr:.4f}")

if matches:
    vx = int(matches[0].x - tmpl_cv.shape[1]/2)
    vy = int(matches[0].y - tmpl_cv.shape[0]/2)
    vulkan_patch = img_cv[vy:vy+tmpl_cv.shape[0], vx:vx+tmpl_cv.shape[1]]
    vulkan_patch_corr = np.corrcoef(vulkan_patch.flatten(), tmpl_cv.flatten())[0,1]
    print(f"Template vs Vulkan patch correlation: {vulkan_patch_corr:.4f}")
