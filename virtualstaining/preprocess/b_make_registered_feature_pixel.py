import os
import shutil
import cv2
import pyvips
import openslide
import pickle
import glob
from utils.label_mapping import biomarkers_dict
import re
import h5py
import numpy as np
from datasets.dataset_coords import marker_dataset_coords
import torch
import os
import shutil
import tqdm
from models.convnextv2 import convnextv2_T
import torch

def b_run_make_registered(
        convnextv2_pth="/path/to/backbone_pretrained.pt",
        target_markers=('DAPI', 'SMA', 'CD3e', 'CD20', 'CD141', 'Pan-Cytokeratin', 'Ki67', 'CD15', 'CD68'),
        root=r"/path/to/mif_registered_data_mifref",
        croods_root="/path/to/coords256_overlap128/patches",
        save_pth="/path/to/he_mif_features",
        batch_size=256,
):
    print(biomarkers_dict)
    id_list = glob.glob(f"{root}/*HE*.ome.tiff")
    id_list = [re.findall(r"_\d+_Scan", x)[0] for x in id_list]
    id_list.sort()
    print(id_list, flush=True)

    # shutil.rmtree(save_pth, ignore_errors=True)
    os.makedirs(save_pth, exist_ok=True)

    print("batch_size", batch_size)
    model = convnextv2_T(convnextv2_pth)
    model.eval()
    model.cuda()

    for wsi_id in id_list:
        he_pth = glob.glob(f"{root}/*HE*{wsi_id}*.ome.tiff")[0]
        mif_pth = glob.glob(f"{root}/*CODEX*{wsi_id}*.ome.tiff")[0]
        print(he_pth)
        print(mif_pth, flush=True)
        he_nm = he_pth.split("/")[-1].split('\\')[-1].replace(".ome.tiff", "")
        coords_pth = f"{croods_root}/{he_nm}.h5"

        inference_dataset = marker_dataset_coords(he_pth, mif_pth, coords_pth, target_markers=target_markers)
        print(len(inference_dataset))
        inference_loader = torch.utils.data.DataLoader(inference_dataset, num_workers=8, batch_size=batch_size,
                                                       persistent_workers=True, pin_memory=True)
        nn = len(inference_loader)
        print("len(inference_loader)", nn)
        resdict_384 = {"features": [],
                       "coords": []}
        resdict_768 = {"features": [],
                       "coords": []}
        for tmp_marker in target_markers:
            resdict_384[tmp_marker] = []
        for tmp_marker in target_markers:
            resdict_768[tmp_marker] = []

        for idx, (region_rgb, mif_list_8, mif_list_16, coords) in enumerate(inference_loader):
            if idx % 20 == 0:
                print(f"{idx}/{nn}", flush=True)
            # print("coords", coords.shape)
            # print(region_rgb.shape)

            with torch.inference_mode():
                region_rgb = region_rgb.cuda()
                tmp_feature = model(region_rgb)
                # 384
                tmp_feature_target = tmp_feature[2].detach().cpu()
                tmp_feature_target = tmp_feature_target[:, :, 4:12, 4:12]
                resdict_384["features"].append(tmp_feature_target)
                resdict_384["coords"].append(coords)
                for idx1, tmp_marker in enumerate(mif_list_16):
                    tmp_marker = tmp_marker.to(torch.uint8)
                    tmp_marker = tmp_marker[:, :, 4:12, 4:12]
                    # print("tmp_marker 384", tmp_marker.shape)
                    resdict_384[target_markers[idx1]].append(tmp_marker)
                # 768
                tmp_feature_target = tmp_feature[3].detach().cpu()
                tmp_feature_target = tmp_feature_target[:, :, 2:6, 2:6]
                # print('tmp_feature_target 768', tmp_feature_target.shape)
                resdict_768["features"].append(tmp_feature_target)
                resdict_768["coords"].append(coords)
                for idx1, tmp_marker in enumerate(mif_list_16):
                    tmp_marker = tmp_marker.to(torch.uint8)
                    tmp_marker = tmp_marker[:, :, 2:6, 2:6]
                    resdict_768[target_markers[idx1]].append(tmp_marker)

        resdict_384["coords"] = torch.concatenate(resdict_384["coords"], dim=0)
        resdict_384["features"] = torch.concatenate(resdict_384["features"], dim=0)
        resdict_768["coords"] = torch.concatenate(resdict_768["coords"], dim=0)
        resdict_768["features"] = torch.concatenate(resdict_768["features"], dim=0)
        for tmp_marker in target_markers:
            resdict_384[tmp_marker] = torch.concatenate(resdict_384[tmp_marker], dim=0)
            resdict_768[tmp_marker] = torch.concatenate(resdict_768[tmp_marker], dim=0)
        print(resdict_384["features"].shape, resdict_384["coords"].shape,
              resdict_384[target_markers[0]].shape, flush=True)
        print(resdict_768["features"].shape, resdict_768["coords"].shape,
              resdict_768[target_markers[0]].shape, flush=True)
        print(resdict_768.keys())
        torch.save(resdict_384, f"{save_pth}/{wsi_id}_384.pt")
        torch.save(resdict_768, f"{save_pth}/{wsi_id}_768.pt")
        print(f"{wsi_id} end\n", flush=True)


        print(flush=True)

    print('b run_make_registered pixel end', flush=True)


if __name__ == "__main__":
    b_run_make_registered()
