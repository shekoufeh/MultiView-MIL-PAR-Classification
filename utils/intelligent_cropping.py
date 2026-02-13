import os
import glob
import cv2
import torch
import numpy as np
import argparse
from sklearn.cluster import KMeans
from tqdm import tqdm

# --- HYPERPARAMETERS ---
MAX_CLUSTERS = 5
MIN_CLUSTERS = 3
MIN_WIDTH_RATIO = 0.60
PADDING_PCT = 0.05
INFERENCE_SIZE = (700, 700)
FINAL_SIZE = (700, 700)

DEBUG_COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
]

def get_teeth_bbox_from_clusters(features, original_img_small, filename, n_clusters, save_debug, debug_prefix):
    """
    Core Logic: Clusters -> Symmetry Merge -> Heuristics -> BBox
    """
    B, N, D = features.shape
    h_grid = int(np.sqrt(N))
    w_grid = int(np.sqrt(N))
    
    flat_features = features.squeeze(0).cpu().numpy()
    
    # 1. K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(flat_features)
    label_map = labels.reshape(h_grid, w_grid)

    # 1.5 Symmetry Merge for Occlusal views
    fname_lower = filename.lower()
    is_occlusal = "_up" in fname_lower or "_down" in fname_lower
    
    if is_occlusal:
        mid_col = w_grid // 2
        left_half = label_map[:, :mid_col]
        right_half = label_map[:, mid_col:]
        
        if right_half.shape[1] > left_half.shape[1]:
            right_half = right_half[:, :left_half.shape[1]]
        elif left_half.shape[1] > right_half.shape[1]:
            left_half = left_half[:, -right_half.shape[1]:]

        left_flipped = np.fliplr(left_half)
        mapping = {}
        
        for l_id in range(n_clusters):
            l_mask = (left_flipped == l_id)
            if np.sum(l_mask) > 0:
                corresponding_pixels = right_half[l_mask]
                if corresponding_pixels.size > 0:
                    counts = np.bincount(corresponding_pixels.flatten(), minlength=n_clusters)
                    best_r_id = np.argmax(counts)
                    mapping[l_id] = best_r_id
        
        for r in range(h_grid):
            for c in range(mid_col):
                old_id = label_map[r, c]
                if old_id in mapping:
                    label_map[r, c] = mapping[old_id]

    gray_img = cv2.cvtColor(original_img_small, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray_img.shape[:2]

    # Debug: Save Cluster Map
    if save_debug and debug_prefix:
        label_map_vis = cv2.resize(label_map, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        color_vis = np.zeros((h_img, w_img, 3), dtype=np.uint8)
        for k in range(n_clusters):
            color_idx = k % len(DEBUG_COLORS)
            color_vis[label_map_vis == k] = DEBUG_COLORS[color_idx]
        vis_combined = np.hstack([original_img_small, color_vis])
        cv2.imwrite(f"{debug_prefix}_k{n_clusters}_01_clusters.jpg", vis_combined)
    
    # 2. Heuristics
    box_pct = 0.85 if is_occlusal else 0.40
    center_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cy, cx = h_img // 2, w_img // 2
    box_h, box_w = int(h_img * box_pct), int(w_img * box_pct)
    cv2.rectangle(center_mask, (cx - box_w//2, cy - box_h//2), (cx + box_w//2, cy + box_h//2), 255, -1)

    best_cluster_id, best_score = -1, -1
    for k in range(n_clusters):
        mask_small = (label_map == k).astype(np.uint8)
        mask_full = cv2.resize(mask_small, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        mean_brightness = cv2.mean(gray_img, mask=mask_full)[0]
        cluster_pixel_count = cv2.countNonZero(mask_full)
        
        if cluster_pixel_count == 0:
            centrality = 0
        else:
            intersection = cv2.bitwise_and(mask_full, mask_full, mask=center_mask)
            centrality = cv2.countNonZero(intersection) / cluster_pixel_count
        
        score = mean_brightness * (centrality ** 0.2) if is_occlusal else mean_brightness * centrality

        if score > best_score:
            best_score, best_cluster_id = score, k

    return (label_map == best_cluster_id).astype(np.uint8) * 255

def apply_crop_and_save(img_path, save_path, model, patch_size, save_debug, debug_dir):
    try:
        file_name = os.path.basename(img_path)
        base_name = os.path.splitext(file_name)[0]
        debug_prefix = os.path.join(debug_dir, base_name) if save_debug else None

        original_img = cv2.imread(img_path)
        if original_img is None: return
        h_orig, w_orig, _ = original_img.shape
        
        # Preprocess for DINO
        target_h = (INFERENCE_SIZE[0] // patch_size) * patch_size
        target_w = (INFERENCE_SIZE[1] // patch_size) * patch_size
        img_resized = cv2.resize(original_img, (target_w, target_h))
        
        img_norm = img_resized / 255.0
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        input_tensor = torch.tensor(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().cuda()
        
        with torch.no_grad():
            features_dict = model.forward_features(input_tensor)
            features = features_dict.get("x_norm_patchtokens", list(features_dict.values())[0])

        # Step-Down Adaptive Loop
        selected_box_data, final_k = None, MAX_CLUSTERS
        for k in range(MAX_CLUSTERS, MIN_CLUSTERS - 1, -1):
            teeth_mask_small = get_teeth_bbox_from_clusters(features, img_resized, file_name, k, save_debug, debug_prefix)
            teeth_mask = cv2.resize(teeth_mask_small, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            
            # Cleaning
            teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_OPEN, np.ones((8,8), np.uint8))
            teeth_mask = cv2.morphologyEx(teeth_mask, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
            
            points = cv2.findNonZero(teeth_mask)
            width_ratio, current_box = 0.0, None
            
            if points is not None:
                bx, by, bw, bh = cv2.boundingRect(points)
                if (bw * bh) > (w_orig * h_orig * 0.01):
                    current_box = (bx, by, bw, bh)
                    width_ratio = bw / w_orig
            
            if width_ratio >= MIN_WIDTH_RATIO:
                selected_box_data, final_k = current_box, k
                break
            else:
                selected_box_data, final_k = current_box, k

        # Apply Final Crop
        if selected_box_data:
            x, y, w, h = selected_box_data
        else:
            x, y, w, h = int(w_orig*0.1), int(h_orig*0.1), int(w_orig*0.8), int(h_orig*0.8)

        # Pad and Resize
        pad_x, pad_y = int(w * PADDING_PCT), int(h * PADDING_PCT)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(w_orig, x + w + pad_x), min(h_orig, y + h + pad_y)
        
        crop = original_img[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        dim = max(ch, cw)
        top, bottom = (dim - ch) // 2, dim - ch - (dim - ch) // 2
        left, right = (dim - cw) // 2, dim - cw - (dim - cw) // 2
        
        final_img = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        final_img = cv2.resize(final_img, FINAL_SIZE, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(save_path, final_img)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Intelligent Cropping Script using DINOv3 Features")
    parser.add_argument("--repo_dir", required=True, help="Local path to the DINOv3 repository")
    parser.add_argument("--weights", required=True, help="Path to the DINOv3 model weights (.pth)")
    parser.add_argument("--input_dir", required=True, help="Directory containing source images")
    parser.add_argument("--output_dir", required=True, help="Directory to save cropped images")
    parser.add_argument("--debug", action="store_true", help="Save intermediate debug visualizations")
    args = parser.parse_args()

    # Setup Directories
    os.makedirs(args.output_dir, exist_ok=True)
    debug_dir = os.path.join(args.output_dir, "debug_visuals")
    if args.debug:
        os.makedirs(debug_dir, exist_ok=True)

    # Load Model
    print(f"Loading Model from {args.weights}...")
    model = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.weights)
    model.cuda().eval()
    patch_size = model.patch_size

    files = glob.glob(os.path.join(args.input_dir, "**", "*.png"), recursive=True)
    print(f"Found {len(files)} images. Starting processing...")

    for file_path in tqdm(files):
        fname_lower = os.path.basename(file_path).lower()
        # Logic: Skip occlusal views as per original script requirement
        if "_up" in fname_lower or "_down" in fname_lower:
            continue

        rel_path = os.path.relpath(file_path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        
        if os.path.exists(out_path):
            continue

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        apply_crop_and_save(file_path, out_path, model, patch_size, args.debug, debug_dir)

if __name__ == "__main__":
    main()