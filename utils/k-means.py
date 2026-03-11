import os
import glob
import cv2
import torch
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


CONFIG = {

    "input_feature_dir": "",

    
    "input_rgb_dir": "",


    "yuhou1_gray_dir": "",
    "yuhou2_gray_dir": "",
    "fenxing_gray_dir": "",


    "output_dir_A": "",
    "output_dir_B": "",
    

    "global_model_dir": "",

  
    "n_clusters": 15,
    "pca_dim": 128,

    "bg_threshold": 6,
    "bg_value": 10,

  
    "fenxing_roi_ratio": 0.3,


    "yuhou_window_size": 70,
    "extreme_ratio": 0.1,

    "yuhou_valid_threshold": 0,

   
    "fg_range": (25, 255),


    "max_samples_per_slide": 10000,  
    "global_max_samples": 500000,    
  
    "kmeans_n_init": 10,
    "random_state": 42,
}


def generate_cluster_values(n_clusters, v_min, v_max):
    
    return np.linspace(v_min, v_max, n_clusters).astype(np.uint8)


def load_gray_resize(path, H, W):
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def load_rgb_resize(path, H, W):
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
    return img


def compute_fg_mask_by_std(rgb_img, bg_threshold):
    
    std_map = np.std(rgb_img.astype(np.float32), axis=2)
    fg_mask = std_map > bg_threshold
    return fg_mask


def compute_window_means(gray_img, valid_mask, window_size):
  
    H, W = gray_img.shape
    n_h = (H + window_size - 1) // window_size
    n_w = (W + window_size - 1) // window_size
    
    window_means = np.full((n_h, n_w), -1.0, dtype=np.float32)
    window_coords = []
    
    for i in range(n_h):
        h_start = i * window_size
        h_end = min(h_start + window_size, H)
        
        for j in range(n_w):
            w_start = j * window_size
            w_end = min(w_start + window_size, W)
            
            window_coords.append((i, j, h_start, h_end, w_start, w_end))
            
            win_gray = gray_img[h_start:h_end, w_start:w_end]
            win_valid = valid_mask[h_start:h_end, w_start:w_end]
            
            valid_vals = win_gray[win_valid]
            if valid_vals.size > 0:
                window_means[i, j] = np.mean(valid_vals)
    
    return window_means, window_coords


def select_extreme_windows(window_means, ratio):
    
    valid_mask = window_means >= 0
    if not valid_mask.any():
        return []
    
    valid_means = window_means[valid_mask]
    n_valid = valid_means.size
    
    n_extreme = max(1, int(np.ceil(n_valid * ratio)))
    
    sorted_indices_1d = np.argsort(valid_means)
    
    extreme_indices_1d = np.concatenate([
        sorted_indices_1d[:n_extreme],
        sorted_indices_1d[-n_extreme:]
    ])
    
    extreme_indices_1d = np.unique(extreme_indices_1d)
    
    valid_coords = np.argwhere(valid_mask)
    selected_indices = [tuple(valid_coords[idx]) for idx in extreme_indices_1d]
    
    return selected_indices


def create_fenxing_window_mask(fx_img, roi_threshold, window_size):

    H, W = fx_img.shape
    n_h = (H + window_size - 1) // window_size
    n_w = (W + window_size - 1) // window_size
    
    fenxing_roi_pixel = (fx_img >= roi_threshold)
    

    fenxing_window_mask = np.zeros((H, W), dtype=bool)
    
    for i in range(n_h):
        h_start = i * window_size
        h_end = min(h_start + window_size, H)
        
        for j in range(n_w):
            w_start = j * window_size
            w_end = min(w_start + window_size, W)
            
        
            window_roi = fenxing_roi_pixel[h_start:h_end, w_start:w_end]
            
           
            if window_roi.any():
                fenxing_window_mask[h_start:h_end, w_start:w_end] = True
    
    return fenxing_window_mask

def create_window_mask(H, W, selected_indices, window_coords, valid_mask):
  
    window_mask = np.zeros((H, W), dtype=bool)
    
    window_dict = {(info[0], info[1]): info for info in window_coords}
    
    for i, j in selected_indices:
        if (i, j) not in window_dict:
            continue
        _, _, h_start, h_end, w_start, w_end = window_dict[(i, j)]
        window_mask[h_start:h_end, w_start:w_end] = True
    
    final_mask = window_mask & valid_mask
    
    return final_mask


def extreme_window_mask_from_gray(gray_img, valid_mask, window_size, ratio):

    H, W = gray_img.shape
    
    window_means, window_coords = compute_window_means(gray_img, valid_mask, window_size)
    selected_indices = select_extreme_windows(window_means, ratio)
    extreme_mask = create_window_mask(H, W, selected_indices, window_coords, valid_mask)
    
    return extreme_mask


def create_block_mask(H, W, selected_indices, window_coords):

    block_mask = np.zeros((H, W), dtype=bool)

    window_dict = {(info[0], info[1]): info for info in window_coords}
    
    for i, j in selected_indices:
        if (i, j) not in window_dict:
            continue
        _, _, h_start, h_end, w_start, w_end = window_dict[(i, j)]
      
        block_mask[h_start:h_end, w_start:w_end] = True
    
    return block_mask


def compute_masks_for_slide(pt_path, config, mode='A'):

    filename = os.path.basename(pt_path)
    common_name = os.path.splitext(filename)[0]


    try:
        feature_tensor = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        print(f"[Skip] {common_name}: Failed to load feature - {e}")
        return None
        
    if feature_tensor.ndim != 3:
        print(f"[Skip] {common_name}: feature tensor ndim != 3, got {feature_tensor.shape}")
        return None

    feature_map = feature_tensor.permute(1, 2, 0).contiguous().numpy()
    H, W, C = feature_map.shape


    rgb_path = os.path.join(config["input_rgb_dir"], common_name + ".png")
    rgb_img = load_rgb_resize(rgb_path, H, W)
    if rgb_img is None:
        rgb_path_jpg = os.path.join(config["input_rgb_dir"], common_name + ".jpg")
        rgb_img = load_rgb_resize(rgb_path_jpg, H, W)

    if rgb_img is None:
        print(f"[Skip] {common_name}: RGB image not found")
        return None

    fg_mask = compute_fg_mask_by_std(rgb_img, config["bg_threshold"])

   
    select_mask = None
    
    if mode == 'A':
        y_path = os.path.join(config["yuhou1_gray_dir"], common_name + ".png")
        fx_path = os.path.join(config["fenxing_gray_dir"], common_name + ".png")
        
        y = load_gray_resize(y_path, H, W)
        fx = load_gray_resize(fx_path, H, W)
        
        if y is None or fx is None:
            print(f"[Skip] {common_name}: yuhou1 or fenxing gray not found")
            return None
        

        pixel_calc_mask = fg_mask & (y > config["yuhou_valid_threshold"])
        
    
        window_means, window_coords = compute_window_means(
            y, pixel_calc_mask, config["yuhou_window_size"]
        )
        

        roi_thr = int(round(config["fenxing_roi_ratio"] * 255))
        
 
        filtered_means = window_means.copy()
        
 
        for idx in range(len(window_coords)):
            i, j, h_s, h_e, w_s, w_e = window_coords[idx]
            
            
            if filtered_means[i, j] < 0:
                continue
            
         
            fx_crop = fx[h_s:h_e, w_s:w_e]
            
            
            if not np.any(fx_crop >= roi_thr):
                filtered_means[i, j] = -1.0 
        
       
        selected_indices = select_extreme_windows(filtered_means, config["extreme_ratio"])
        
     
        block_mask = create_block_mask(H, W, selected_indices, window_coords)
        
 
        select_mask = block_mask & fg_mask
        
    elif mode == 'B':
        
        y_path = os.path.join(config["yuhou2_gray_dir"], common_name + ".png")
        y = load_gray_resize(y_path, H, W)
        
        if y is None:
            print(f"[Skip] {common_name}: yuhou2 gray not found")
            return None
      
        pixel_calc_mask = fg_mask & (y > config["yuhou_valid_threshold"])
        
    
        window_means, window_coords = compute_window_means(
            y, pixel_calc_mask, config["yuhou_window_size"]
        )
        
   
        selected_indices = select_extreme_windows(window_means, config["extreme_ratio"])
        
     
        block_mask = create_block_mask(H, W, selected_indices, window_coords)
        
    
        select_mask = block_mask & fg_mask

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return feature_map, select_mask, common_name


def collect_global_features(pt_files, config, mode='A'):

    features_list = []
    total_samples = 0
    

    for pt_path in tqdm(pt_files, desc="收集特征"):
        result = compute_masks_for_slide(pt_path, config, mode)
        if result is None:
            continue
            
        feature_map, select_mask, common_name = result

        selected_features = feature_map[select_mask]  # (N, C)
        n_samples = selected_features.shape[0]
        
        if n_samples == 0:
            print(f"[Warning] {common_name}: 没有选中的像素")
            continue
   
        if n_samples > config["max_samples_per_slide"]:
            indices = np.random.choice(
                n_samples, 
                config["max_samples_per_slide"], 
                replace=False
            )
            selected_features = selected_features[indices]
            n_samples = config["max_samples_per_slide"]
        
        features_list.append((selected_features, common_name))
        total_samples += n_samples

    return features_list, total_samples


def train_global_clustering(features_list, total_samples, config, mode='A'):

   
    if total_samples > config["global_max_samples"]:
        
        sample_ratio = config["global_max_samples"] / total_samples
        
        sampled_features = []
        for features, name in features_list:
            n = features.shape[0]
            n_sample = max(1, int(n * sample_ratio))
            indices = np.random.choice(n, n_sample, replace=False)
            sampled_features.append(features[indices])
        
        all_features = np.vstack(sampled_features)
    else:
   
        all_features = np.vstack([f for f, _ in features_list])
    

    n_samples, n_features = all_features.shape
    actual_pca_dim = min(config["pca_dim"], n_samples, n_features)
    
   
    pca_model = PCA(
        n_components=actual_pca_dim, 
        random_state=config["random_state"]
    )
    X_pca = pca_model.fit_transform(all_features)
    
    explained_var = np.sum(pca_model.explained_variance_ratio_)

    n_clusters = config["n_clusters"]
    
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        random_state=config["random_state"],
        n_init=config["kmeans_n_init"],
        verbose=1
    )
    kmeans_model.fit(X_pca)
    

    model_dir = config["global_model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    
    pca_path = os.path.join(model_dir, f"pca_model_{mode}.pkl")
    kmeans_path = os.path.join(model_dir, f"kmeans_model_{mode}.pkl")
    
    with open(pca_path, 'wb') as f:
        pickle.dump(pca_model, f)
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans_model, f)
    

    return pca_model, kmeans_model


# ==================== 全局聚类：阶段3 - 映射回切片 ====================

def apply_global_clustering(pt_files, pca_model, kmeans_model, config, mode='A'):

    
    output_dir = config["output_dir_A"] if mode == 'A' else config["output_dir_B"]
    os.makedirs(output_dir, exist_ok=True)
    
    pixel_values = generate_cluster_values(
        config["n_clusters"], 
        *config["fg_range"]
    )
    
    success_count = 0
    
    for pt_path in tqdm(pt_files, desc="生成结果"):
        result = compute_masks_for_slide(pt_path, config, mode)
        if result is None:
            continue
        
        feature_map, select_mask, common_name = result
        H, W, C = feature_map.shape
        

        selected_features = feature_map[select_mask]
        n_samples = selected_features.shape[0]
        
  
        X_pca = pca_model.transform(selected_features)
        labels = kmeans_model.predict(X_pca)

        output = np.full((H, W), config["bg_value"], dtype=np.uint8)
        output[select_mask] = pixel_values[labels].astype(np.uint8)
     
        save_path = os.path.join(output_dir, common_name + ".png")
        cv2.imwrite(save_path, output)
        success_count += 1
    
 
def main():
 
    pt_files = sorted(glob.glob(os.path.join(CONFIG["input_feature_dir"], "*.pt")))


    features_A, total_A = collect_global_features(pt_files, CONFIG, mode='A')
    
    if len(features_A) == 0:
        print("[Error]")
    else:
       
        pca_A, kmeans_A = train_global_clustering(features_A, total_A, CONFIG, mode='A')
        
      
        apply_global_clustering(pt_files, pca_A, kmeans_A, CONFIG, mode='A')
    

    features_B, total_B = collect_global_features(pt_files, CONFIG, mode='B')
    
    if len(features_B) == 0:
        print("[Error] ")
    else:
      
        pca_B, kmeans_B = train_global_clustering(features_B, total_B, CONFIG, mode='B')
        
       
        apply_global_clustering(pt_files, pca_B, kmeans_B, CONFIG, mode='B')
    


if __name__ == "__main__":

    np.random.seed(CONFIG["random_state"])
    main()