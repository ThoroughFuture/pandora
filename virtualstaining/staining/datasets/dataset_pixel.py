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

class marker_dataset_pixel(Dataset):
    def __init__(self, root, mode="train", args=None, test_key="_1_Scan"):
        if mode == "train":
            rank = dist.get_rank()
        else:
            rank = 0
        # rank = 0

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
        add_idx = []
        tmp_add_index = 0
        for slide_nm in self.slide_list:
            add_idx.append(tmp_add_index)
            tmp_feature_dict = f"{feature_dir}/{slide_nm}.pt"
            tmp_dict = torch.load(tmp_feature_dict, weights_only=False)
            tmp_feature = tmp_dict["features"]
            if rank == 0:
                print(slide_nm)
                print("tmp_feature before", tmp_feature.shape)
            if len(tmp_feature.shape) > 2:
                tmp_feature = tmp_feature.moveaxis(1, -1)
                if rank == 0:
                    print("tmp_feature mid", tmp_feature.shape)
                tmp_feature = tmp_feature.reshape((-1, tmp_feature.shape[-1]))
            if rank == 0:
                print("tmp_feature after", tmp_feature.shape)
                print(flush=True)
            tmp_add_index += tmp_feature.shape[0]
            self.feature_all.append(tmp_feature)

        self.feature_all = torch.concatenate(self.feature_all, dim=0)

        self.pos_dict = {}
        self.neg_dict = {}
        for marker in self.maker_list:
            self.pos_dict[marker] = []
        for marker in self.maker_list:
            for idx, slide_nm in enumerate(self.slide_list):
                tmp_index_pth = f"{root}/pos/{marker}/{slide_nm}.pt"
                tmp_idx = torch.load(tmp_index_pth, weights_only=False)
                # if rank == 0:
                #     print(f"{slide_nm} tmp_idx pos", np.max(tmp_idx))
                tmp_idx += add_idx[idx]
                self.pos_dict[marker].append(tmp_idx)
            self.pos_dict[marker] = np.concatenate(self.pos_dict[marker])
        for marker in self.maker_list:
            self.neg_dict[marker] = []
        for marker in self.maker_list:
            for idx, slide_nm in enumerate(self.slide_list):
                tmp_index_pth = f"{root}/neg/{marker}/{slide_nm}.pt"
                tmp_idx = torch.load(tmp_index_pth, weights_only=False)
                # if rank == 0:
                #     print(f"{slide_nm} tmp_idx neg", np.max(tmp_idx))
                tmp_idx += add_idx[idx]
                self.neg_dict[marker].append(tmp_idx)
            self.neg_dict[marker] = np.concatenate(self.neg_dict[marker])

        if rank == 0:
            print(mode, len(self.feature_all))
            print(self.feature_all.shape)
            print(np.max(self.neg_dict["DAPI"]))

    def __len__(self):
        return len(self.feature_all)

    def __getitem__(self, idx):
        feature_list = []
        label_list = []
        for _, tmp_marker in enumerate(self.maker_list):
            random_num = random.random()
            if random_num <= self.pos_rate:
                tmp_dict = self.pos_dict
                label_list.append(1)
            else:
                tmp_dict = self.neg_dict
                label_list.append(0)
            feature_list.append(self.feature_all[tmp_dict[tmp_marker][idx % len(tmp_dict[tmp_marker])]])

        return feature_list, label_list


if __name__ == "__main__":
    root = [
        r"/home/wujunxian/xinyi_features/pixel_data",
    ]

    dataset = marker_dataset_pixel(root, mode="train")
    print(len(dataset))
    batch_size = 2
    loader = torch.utils.data.DataLoader(dataset, num_workers=3, batch_size=batch_size,
                                         persistent_workers=True, pin_memory=True)

    # for i, (x, marker_mask) in enumerate(loader):
    #     # print(y)
    #     print(x.shape)
    #     for x in marker_mask:
    #         print(x.shape)
    #
    #     print()
    #     break
