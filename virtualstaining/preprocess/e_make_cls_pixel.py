import shutil
import torch
import os
import glob
import gc
import numpy as np
import cv2
import re
import os

def remove_number_suffix(filename):
    pattern = r'_\d+(?=\.|$)'
    new_filename = re.sub(pattern, '', filename)
    return new_filename


def make01_via_cutoff(pt_dict, otsu_cutoff_dict, target_marker="DAPI"):
    tmp_cutoff = otsu_cutoff_dict[target_marker]
    tmp_pixel = pt_dict[target_marker].numpy()
    tmp_pixel = np.moveaxis(tmp_pixel, 1, -1)
    tmp_pixel = tmp_pixel.flatten()
    tmp_feature = pt_dict["features"].numpy()
    tmp_cls = np.zeros(tmp_pixel.shape)
    tmp_cls[tmp_pixel >= tmp_cutoff] = 1
    return tmp_feature, tmp_cls


def e_run_cls(
        cutoff_root="/path/to/pixel_cutoff_otsu",
        pt_root="/path/to/pixel_data",
        save_pth="/path/to/pixel_dataset",
        marker_list=('DAPI', 'SMA', 'CD3&CD20', 'CD141', 'Pan-Cytokeratin', 'Ki67', 'CD15', 'CD68', "CD3e", "CD20"),
        target_dim=768,
):
    pt_list = glob.glob(f"{pt_root}/*_{str(target_dim)}.pt")
    if str(target_dim) not in save_pth:
        save_pth = f"{save_pth.rstrip('/')}_{str(target_dim)}"
    print('save_pth', save_pth)
    shutil.rmtree(save_pth, ignore_errors=True)
    os.makedirs(f"{save_pth}/features/", exist_ok=True)
    for marker in marker_list:
        os.makedirs(f"{save_pth}/pos/{marker}/", exist_ok=True)
        os.makedirs(f"{save_pth}/neg/{marker}/", exist_ok=True)

    otsu_cutoff_dict_pth = f"{cutoff_root}/pixel_cutoff_otsu_{str(target_dim)}.pt"
    otsu_cutoff_dict = torch.load(otsu_cutoff_dict_pth)
    otsu_cutoff_split_dict_pth = f"{cutoff_root}/pixel_cutoff_otsu_split_{str(target_dim)}.pt"
    otsu_cutoff_split_dict = torch.load(otsu_cutoff_split_dict_pth)
    print(otsu_cutoff_dict)
    print(otsu_cutoff_split_dict, flush=True)
    my_cutoff_dict = otsu_cutoff_dict

    for pt_pth in pt_list:
        tmp_nm = pt_pth.split(os.sep)[-1].replace(".pt", "")
        tmp_nm = remove_number_suffix(tmp_nm)
        pt_dict = torch.load(pt_pth, weights_only=False)
        if not os.path.exists(f"{save_pth}/features/{tmp_nm}.pt"):
            shutil.copy(pt_pth, f"{save_pth}/features/{tmp_nm}.pt")
        for marker in marker_list:
            print(tmp_nm, marker)
            if marker == "CD3&CD20":
                tmp_feature, tmp_cls_cd3 = make01_via_cutoff(pt_dict, my_cutoff_dict, target_marker="CD3e")
                _, tmp_cls_cd20 = make01_via_cutoff(pt_dict, my_cutoff_dict, target_marker="CD20")
                tmp_cls = np.logical_or(tmp_cls_cd3, tmp_cls_cd20)

                indices_pos = np.where(tmp_cls == 1)[0]
                indices_neg = np.where(tmp_cls == 0)[0]
                print("pos", indices_pos.shape)
                print("neg", indices_neg.shape, flush=True)

            else:
                tmp_feature, tmp_cls = make01_via_cutoff(pt_dict, my_cutoff_dict, target_marker=marker)

                indices_pos = np.where(tmp_cls == 1)[0]
                indices_neg = np.where(tmp_cls == 0)[0]
                print("pos", indices_pos.shape)
                print("neg", indices_neg.shape, flush=True)

            if len(indices_neg) == 0:
                print("len(indices_neg) == 0")
                indices_pos, indices_neg = indices_neg, indices_pos
                print("pos reverse", indices_pos.shape)
                print("neg reverse", indices_neg.shape, flush=True)

            torch.save(indices_pos, f"{save_pth}/pos/{marker}/{tmp_nm}.pt")
            torch.save(indices_neg, f"{save_pth}/neg/{marker}/{tmp_nm}.pt")

            print()
    print("e end", flush=True)

if __name__ == "__main__":
    e_run_cls()
