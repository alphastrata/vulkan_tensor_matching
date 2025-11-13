from vulkan_tensor_matching import ImageData, VulkanTensorMatcher
from pathlib import Path
import math, numpy as np

# Check source_6.png: what's at position (669,59)?
src = Path("test_data/extensive/source_images")
img_path = src / "source_6.png"
from PIL import Image
img_pil = Image.open(img_path).convert("L")
img_np = np.array(img_pil).astype(np.float32) / 255.0

# Check all failing images
for name, tl, wrong in [
    ("source_6.png",  (102,262), (669,59)),
    ("source_16.png", (358,354), (421,254)),
    ("source_4.png",  (658,270), (576,59)),
]:
    img_path = src / name
    img = Image.open(img_path).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0

    t = arr[tl[1]:tl[1]+64, tl[0]:tl[0]+64]
    w = arr[wrong[1]:wrong[1]+64, wrong[0]:wrong[0]+64]

    print(f"\n{name}:")
    print(f"  GT   ({tl[0]},{tl[1]}): var={t.var():.6f}, std={t.std():.4f}, mean={t.mean():.4f}")
    print(f"  Wrong({wrong[0]},{wrong[1]}): var={w.var():.8f}, std={w.std():.6f}, mean={w.mean():.4f}")

    # Check if wrong patch is almost uniform
    print(f"  Wrong patch min={w.min():.4f} max={w.max():.4f} range={w.max()-w.min():.6f}")

    # CPU NCC
    t_norm = (t - t.mean()) / (t.std() * 64)
    w_std = max(w.std(), 1e-6)
    w_norm = (w - w.mean()) / (w_std * 64)
    ncc = np.sum(t_norm * w_norm)
    print(f"  CPU NCC: {ncc:.6f}")

# Check what all the 'wrong' position have in common
print("\n\nChecking what positions have corr=1.0 in source_6.png:")
arr = np.array(Image.open(src/"source_6.png").convert("L")).astype(np.float32)/255.0
t = arr[262:326, 102:166]
t_norm = (t - t.mean()) / (t.std() * 64)
for wy in range(59, 200, 5):
    for wx in range(0, 800-64, 40):
        p = arr[wy:wy+64, wx:wx+64]
        if p.var() < 1e-4:  # nearly flat
            p_std = max(p.std(), 1e-6)
            p_norm = (p - p.mean()) / (p_std * 64)
            ncc = np.sum(t_norm * p_norm)
            if abs(ncc) > 0.2:
                print(f"  ({wx},{wy}) var={p.var():.2e} ncc={ncc:.4f}")
