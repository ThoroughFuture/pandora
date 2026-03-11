from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
import glob
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import random
import torch.distributed as dist

from PIL import Image
import torchvision.transforms.functional as F


class dataset_pixel_inference(Dataset):
    def __init__(self, root, mode="test", args=None, test_key="_1_Scan",
                 marker_list=('DAPI', 'SMA', 'CD3&CD20', 'CD141', 'Pan-Cytokeratin', 'Ki67', 'CD15', 'CD68')):
        if mode == "train":
            rank = dist.get_rank()
        else:
            rank = 0
        # rank = 0
        self.marker_list = marker_list
        self.pos_rate = 0.3

        self.maker_list = args.marker_list
        if rank == 0:
            print('root', root, flush=True)
            print("pos_rate", self.pos_rate)

        self.size = 256
        if rank == 0:
            print('Start load', flush=True)

        feature_dir = f"{root}/features"
        self.slide_list = os.listdir(feature_dir)
        self.slide_list = [x.replace(".pt", "") for x in self.slide_list]
        if mode == "train":
            self.slide_list = [x for x in self.slide_list if test_key not in x]
        elif mode == "test":
            self.slide_list = [x for x in self.slide_list if test_key in x]

        if rank == 0:
            print(mode, self.slide_list)

        self.feature_all = []
        self.coords_all = []
        self.gt_dict = {}
        for marker in self.marker_list:
            self.gt_dict[marker] = []
        for slide_nm in self.slide_list:
            tmp_feature_dict = f"{feature_dir}/{slide_nm}.pt"
            tmp_dict = torch.load(tmp_feature_dict, weights_only=False)
            tmp_feature = tmp_dict["features"]
            tmp_coords = tmp_dict["coords"]

            if rank == 0:
                print(slide_nm)
                print("tmp_feature", tmp_feature.shape)
                print("tmp_coords", tmp_coords.shape)

            self.feature_all.append(tmp_feature)
            self.coords_all.append(tmp_coords)
            for marker in self.marker_list:
                self.gt_dict[marker].append(tmp_dict[marker])

        self.feature_all = torch.concatenate(self.feature_all, dim=0)
        self.coords_all = torch.concatenate(self.coords_all, dim=0)
        for marker in self.marker_list:
            self.gt_dict[marker] = torch.concatenate(self.gt_dict[marker], dim=0)

        if rank == 0:
            print(mode, len(self.feature_all))
            print(self.feature_all.shape)
            print(self.coords_all.shape)
            print(self.gt_dict[self.marker_list[0]].shape)

    def __len__(self):
        return len(self.feature_all)

    def __getitem__(self, idx):
        feature = self.feature_all[idx]
        coords = self.coords_all[idx]
        markers_gt = []
        for marker in self.marker_list:
            markers_gt.append(self.gt_dict[marker][idx])

        return feature, coords, markers_gt
