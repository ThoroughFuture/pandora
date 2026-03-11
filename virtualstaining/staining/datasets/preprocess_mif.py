import numpy
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2
import os
import numpy as np
from datasets.pooling_funcs import make_max_pooling, make_avg_pooling


class preprocess_mif:
    def __init__(self, alpha=0.5, in_dim=256, out_dim=16, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.alpha = alpha
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.max_pool_4times = make_max_pooling(self.in_dim, self.out_dim)
        self.avg_pool_4times = make_avg_pooling(self.in_dim, self.out_dim)

    def __call__(self, mif_patch):
        mif_pic = self.transform(mif_patch)
        mif_pic_max = self.max_pool_4times(mif_pic)
        mif_pic_avg = self.avg_pool_4times(mif_pic)
        mif_out = self.alpha * mif_pic_max + (1 - self.alpha) * mif_pic_avg
        return mif_out

class preprocess_mif_wotransform:
    def __init__(self, alpha=0.5, in_dim=256, out_dim=16):
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.alpha = alpha
        self.max_pool_4times = make_max_pooling(self.in_dim, self.out_dim)
        self.avg_pool_4times = make_avg_pooling(self.in_dim, self.out_dim)

    def __call__(self, mif_patch):
        mif_patch = torch.tensor(np.array(mif_patch), dtype=torch.float32).unsqueeze(0)
        mif_pic_max = self.max_pool_4times(mif_patch)
        mif_pic_avg = self.avg_pool_4times(mif_patch)
        mif_out = self.alpha * mif_pic_max + (1 - self.alpha) * mif_pic_avg
        return mif_out
