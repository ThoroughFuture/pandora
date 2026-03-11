import os
import cv2
import pyvips
import openslide
import pickle
import glob
from utils.label_mapping import biomarkers_dict
import re
from PIL import Image
import numpy as np
import torch


def c_run_merge(root_merge):
    print('c_run_merge', flush=True)
    # 384
    slide_list = glob.glob(f"{root_merge}/*_384.pt")
    print(slide_list)
    print(len(slide_list))
    for slide_pth in slide_list:
        print('slide_pth', slide_pth)
        tmp_dict = torch.load(slide_pth)
        cd3_tmp = tmp_dict["CD3e"]
        cd20_tmp = tmp_dict["CD20"]
        merged_img = torch.maximum(cd3_tmp, cd20_tmp)
        print(merged_img.shape, flush=True)

        tmp_dict["CD3&CD20"] = merged_img
        torch.save(tmp_dict, slide_pth)

    # 768
    slide_list = glob.glob(f"{root_merge}/*_768.pt")
    print(slide_list)
    for slide_pth in slide_list:
        print('slide_pth', slide_pth, flush=True)
        tmp_dict = torch.load(slide_pth)
        cd3_tmp = tmp_dict["CD3e"]
        cd20_tmp = tmp_dict["CD20"]
        merged_img = torch.maximum(cd3_tmp, cd20_tmp)
        print(merged_img.shape, flush=True)

        tmp_dict["CD3&CD20"] = merged_img
        torch.save(tmp_dict, slide_pth)

    print("c end")


if __name__ == "__main__":
    c_run_merge()
