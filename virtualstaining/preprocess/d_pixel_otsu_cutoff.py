import glob
import os
import cv2
import torch
import numpy as np
import gc
import shutil

def d_run_otsu_cutoff(
        marker_list=('DAPI', 'SMA', 'CD3&CD20', 'CD141', 'Pan-Cytokeratin', 'Ki67', 'CD15', 'CD68', "CD3e", "CD20"),
        pt_root="/path/to/pixel_data",
        save_pth="/path/to/pixel_cutoff_otsu",
        target_dim=768,
):
    print('marker_list d', marker_list)
    print('target_dim', target_dim, flush=True)
    os.makedirs(save_pth, exist_ok=True)
    pt_list = glob.glob(f"{pt_root}/*_{target_dim}.pt")
    print(pt_list)
    otsu_cutoff = {}
    for marker in marker_list:
        print(marker, flush=True)
        tmp_list = []
        for pt_pth in pt_list:
            pt_dict = torch.load(pt_pth, weights_only=False)
            tmp_img = pt_dict[marker].numpy().astype(np.uint8)
            tmp_list.append(tmp_img)
        img_all = np.concatenate(tmp_list)
        cv2_thresh, _ = cv2.threshold(img_all, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(cv2_thresh, flush=True)
        gc.collect()
        otsu_cutoff[marker] = cv2_thresh
    torch.save(otsu_cutoff, f"{save_pth}/pixel_cutoff_otsu_{target_dim}.pt")
    print()

    print("Excute split", flush=True)
    otsu_cutoff_split = {}
    for marker in marker_list:
        otsu_cutoff_split[marker] = []
    for marker in marker_list:
        print(marker, flush=True)
        for pt_pth in pt_list:
            print(pt_pth.split(os.sep)[-1], flush=True)
            pt_dict = torch.load(pt_pth, weights_only=False)
            tmp_img = pt_dict[marker].numpy().astype(np.uint8)
            cv2_thresh, _ = cv2.threshold(tmp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_cutoff_split[marker].append(cv2_thresh)
            print(cv2_thresh, flush=True)
            gc.collect()
    print(otsu_cutoff_split)
    torch.save(otsu_cutoff_split, f"{save_pth}/pixel_cutoff_otsu_split_{target_dim}.pt")

    print("d end", flush=True)
