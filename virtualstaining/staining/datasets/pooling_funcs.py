import numpy
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2
import os
import numpy as np


def make_max_pooling(in_dim=256, out_dim=16):
    if in_dim % out_dim != 0:
        raise NotImplementedError
    l_num = np.log2(in_dim / out_dim)
    if np.isscalar(l_num) and l_num.is_integer():
        l_num = int(l_num)
    else:
        raise NotImplementedError
    max_pool_4times = nn.Sequential(
        *[nn.MaxPool2d(2, 2, 0) for _ in range(l_num)]
    )
    return max_pool_4times


def make_avg_pooling(in_dim=256, out_dim=16):
    if in_dim % out_dim != 0:
        raise NotImplementedError
    l_num = np.log2(in_dim / out_dim)
    if np.isscalar(l_num) and l_num.is_integer():
        l_num = int(l_num)
    else:
        raise NotImplementedError
    avg_pool_4times = nn.Sequential(
        *[nn.AvgPool2d(2, 2, 0) for _ in range(l_num)]
    )
    return avg_pool_4times
