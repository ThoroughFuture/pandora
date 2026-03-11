from b_make_registered_feature_pixel import b_run_make_registered
from c_merge_CD3CD20_pt import c_run_merge
from d_pixel_otsu_cutoff import d_run_otsu_cutoff
from e_make_cls_pixel import e_run_cls
from utils.label_mapping import biomarkers_dict
import copy
import os

if __name__ == "__main__":
    save_root = "/path/to/root"
    # b
    target_markers_b = tuple(biomarkers_dict.keys())
    print('target_markers_b', target_markers_b)
    root_b = "/path/to/wsi_pth"
    croods_root_b = "/path/to/patches"
    save_pth_b = f"{save_root}/he_mif_features"
    # c
    root_merge_c = save_pth_b
    # d
    pt_root_d = save_pth_b
    save_pth_d = f"{save_root}/pixel_cutoff_otsu"
    target_markers_d = target_markers_b + ("CD3&CD20",)
    # e
    cutoff_root_e = save_pth_d
    pt_root_e = save_pth_b
    save_pth_e = f"{save_root}/pixel_dataset"
    marker_list_e = target_markers_d

    b_run_make_registered(
        target_markers=target_markers_b,
        root=root_b,
        croods_root=croods_root_b,
        save_pth=save_pth_b,
    )

    c_run_merge(root_merge=root_merge_c)

    d_run_otsu_cutoff(
        marker_list=target_markers_d,
        pt_root=pt_root_d,
        target_dim=768,
        save_pth=save_pth_d
    )
    d_run_otsu_cutoff(
        marker_list=target_markers_d,
        pt_root=pt_root_d,
        target_dim=384,
        save_pth=save_pth_d
    )

    e_run_cls(
        cutoff_root=cutoff_root_e,
        pt_root=pt_root_e,
        save_pth=save_pth_e,
        marker_list=marker_list_e,
        target_dim=768,
    )
    e_run_cls(
        cutoff_root=cutoff_root_e,
        pt_root=pt_root_e,
        save_pth=save_pth_e,
        marker_list=marker_list_e,
        target_dim=384,
    )

    print("run Done")
