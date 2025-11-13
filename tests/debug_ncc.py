import numpy as np
from PIL import Image
from pathlib import Path

def debug_ncc():
    # Load images
    img_path = "test_data/lenna.png"
    tmpl_path = "test_data/templates/test1.png"
    
    if not Path(img_path).exists():
        print(f"Error: {img_path} not found")
        return

    print("Loading images...")
    img = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
    tmpl = np.array(Image.open(tmpl_path).convert('L'), dtype=np.float32) / 255.0
    
    print(f"Image: {img.shape}")
    print(f"Template: {tmpl.shape}")
    
    th, tw = tmpl.shape
    ih, iw = img.shape
    
    # Ground truth center approx (45, 95)
    # Top-left should be approx (13, 63)
    gt_x, gt_y = 13, 63
    
    # Extract patch at GT to verify it matches template
    patch_gt = img[gt_y:gt_y+th, gt_x:gt_x+tw]
    print(f"Patch at GT mean: {np.mean(patch_gt):.4f}")
    print(f"Template mean: {np.mean(tmpl):.4f}")
    print(f"Abs Diff: {np.mean(np.abs(patch_gt - tmpl)):.4f}")
    
    # Calculate Template Stats
    tmpl_mean = np.mean(tmpl)
    tmpl_dev = np.std(tmpl)
    tmpl_norm = (tmpl - tmpl_mean) / (tmpl_dev * np.sqrt(tw * th))
    
    # Run Naive NCC at GT location
    patch = img[gt_y:gt_y+th, gt_x:gt_x+tw]
    patch_mean = np.mean(patch)
    patch_dev = np.std(patch)
    
    if patch_dev < 1e-5:
        print("Patch variance too low")
        return
        
    patch_norm = (patch - patch_mean) / (patch_dev * np.sqrt(tw * th))
    
    correlation = np.sum(tmpl_norm * patch_norm)
    print(f"\nCalculated NCC at GT ({gt_x}, {gt_y}): {correlation:.6f}")
    
    # Check a few pixels around
    print("\nScanning region around GT:")
    best_corr = -1.0
    best_loc = (0, 0)
    
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            y = gt_y + dy
            x = gt_x + dx
            
            p = img[y:y+th, x:x+tw]
            pm = np.mean(p)
            pd = np.std(p)
            
            # Manual calculation matching shader logic
            # Shader: sum(tmpl_norm_component * (pixel - patch_mean)) * (1 / (patch_dev * sqrt(count)))
            # Here tmpl_norm is already normalized.
            # Let's match shader exactly:
            # res += tmpl_component * (pixel - patch_mean)
            # final = res * norm_factor
            
            diff_p = p - pm
            res = np.sum(tmpl_norm * diff_p) # tmpl_norm already has 1/(std_t*sqrt(N))
            
            # norm_factor = 1.0 / (patch_dev * sqrt(N))
            final = res * (1.0 / (pd * np.sqrt(tw*th)))
            
            if final > best_corr:
                best_corr = final
                best_loc = (x, y)
                
            if dx == 0 and dy == 0:
                print(f"  Offset (0, 0): {final:.6f} (Should be ~1.0)")

    print(f"\nBest found in region: {best_corr:.6f} at {best_loc}")

if __name__ == "__main__":
    debug_ncc()
