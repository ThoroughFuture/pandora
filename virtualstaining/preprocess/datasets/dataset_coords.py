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
from datasets.preprocess_mif import preprocess_mif_wotransform
import pyvips
import openslide
from utils.label_mapping import all_marker_tuple, biomarkers_dict
import h5py


class marker_dataset_coords(Dataset):
    def __init__(self, he_pth, mif_pth, coords_pth, target_markers):
        self.target_markers = target_markers
        print('target_markers', self.target_markers)
        self.slide_he = openslide.OpenSlide(he_pth)
        self.mif_list = []
        for tmp_marker in self.target_markers:
            slide_mif = pyvips.Image.new_from_file(
                mif_pth,
                access="sequential",
                page=biomarkers_dict[tmp_marker],
                n=1
            )
            self.mif_list.append(slide_mif)

        hdf5_file = h5py.File(coords_pth, "r")
        self.coords = np.array(hdf5_file['coords'])
        # level = hdf5_file['coords'].attrs['patch_level']
        size = hdf5_file['coords'].attrs['patch_size']
        self.size = size
        for name, value in hdf5_file['coords'].attrs.items():
            if name == "save_path":
                continue
            print(name, value)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.preprocess_8 = preprocess_mif_wotransform(in_size=self.size, out_size=8)
        self.preprocess_16 = preprocess_mif_wotransform(in_size=self.size, out_size=16)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x_, y_ = self.coords[idx]
        # HE
        region_rgb = self.slide_he.read_region((x_, y_), 0, (self.size, self.size)).convert("RGB")
        region_rgb = self.transform(region_rgb)
        # mIF list
        mif_list_16 = []
        mif_list_8 = []
        for idx, tmp_mif in enumerate(self.mif_list):
            # tmp_marker = self.target_markers[idx]
            marker_image = tmp_mif.crop(x_, y_, self.size, self.size).numpy()
            mif_list_8.append(self.preprocess_8(marker_image))
            mif_list_16.append(self.preprocess_16(marker_image))

        return region_rgb, mif_list_8, mif_list_16, torch.tensor([x_, y_])

