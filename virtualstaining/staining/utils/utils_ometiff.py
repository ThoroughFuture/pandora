import argparse
import os
import random
import numpy as np
import torch
from utils.utils import get_args
import shutil
import argparse
import logging
import os
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils.utils import *
import gc
from datasets.dataset_pixel import marker_dataset_pixel
import cv2
from PIL import Image
import tifffile as tiff
import imagecodecs


def save_ometiff_from_np(marker_list, img, size, output_path="output_ome.tif"):
    ome_metadata = {
        "PhysicalSizeX": 0.25,
        "PhysicalSizeY": 0.25,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeYUnit": "µm",
        "Channel": [{"@Name": x} for x in marker_list],
        'Image': {
            'Name': 'Test OME-TIFF Image',
            'Dimensions': 'CYX',
            'SizeC': size[0],
            'SizeY': size[1],
            'SizeX': size[2],
            # 'Pixels': {
            #     'Channel': [{'@Name': x} for x in marker_list]
            # }
        }
    }

    tiff.imwrite(
        output_path,
        img,
        ome=True,
        bigtiff=True,
        tile=(256, 256),
        # compression="jpeg",
        compressionargs={"level": 90},
        metadata=ome_metadata,
        dtype=np.uint8
    )
