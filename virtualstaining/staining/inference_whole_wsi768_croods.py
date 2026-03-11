import argparse
import copy
import glob
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
import torch.nn.functional as F
import gc
from datasets.dataset_pixel_inference import dataset_pixel_inference
import cv2
from PIL import Image
import tifffile as tiff
from scipy.stats import spearmanr
import pyvips
from utils.utils import calculate_dice
from utils.utils_ometiff import save_ometiff_from_np
from models.linear_head import linear_head
from utils.label_mapping import all_marker_tuple
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import pandas as pd


def find_best_cutoff_by_youden(y_true, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_cutoff = thresholds[best_idx]
    max_youden = youden[best_idx]
    return best_cutoff, max_youden


def calculate_roc_auc(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc, y_true, y_pred


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int,
                    help='node rank for distributed training')
parser.add_argument('--yml_opt_path', type=str, default='linear_768_marker_all')
args = parser.parse_args()

if __name__ == "__main__":
    args = get_args_inference(args)
    if args.marker_list == "all":
        args.marker_list = all_marker_tuple

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    inference_save_root = "/path/to/inferance_save_pth"
    inference_save_tiff_pth = f"{inference_save_root}/tiff"
    metric_save_pth = f"{inference_save_root}/metrics"
    print('metric_save_pth', metric_save_pth)
    print('inference_save_tiff_pth', inference_save_tiff_pth)

    # shutil.rmtree(inference_save_tiff_pth, ignore_errors=True)
    os.makedirs(inference_save_tiff_pth, exist_ok=True)

    cutoff_pth = args.cutoff_pth
    otsu_cutoff_dict = torch.load(cutoff_pth)
    print(otsu_cutoff_dict, flush=True)
    print('args', args, flush=True)
    marker_list = args.marker_list
    print('marker_list', marker_list)
    args.is_pretrain = True
    args.exp = 'TU'
    snapshot_path = f"{args.snapshot_path}/{args.exp}"
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path

    model = linear_head(marker_num=len(args.marker_list), in_dim=768)
    model.cuda()

    batch_size = 64
    test_key_list = ["_1_Scan", "_14_Scan", "_18_Scan", "_5_Scan"]
    # test_key_list = [args.test_key]
    print('test_key_list', test_key_list)
    for tmp_key_test in test_key_list:
        args.test_key = tmp_key_test
        print("args.test_key", args.test_key)

        snapshot_path_tmp = snapshot_path + '_testkey' + str(args.test_key)
        save_model_pth_tmp = f"{args.save_model_pth}/{snapshot_path_tmp.split(os.sep)[-1]}"
        print("model load")
        linear_pth = f"{save_model_pth_tmp}/last.pth"
        # linear_pth = f"{save_model_pth_tmp}/epoch_1.pth"
        print('linear_pth', linear_pth)
        model.load_state_dict(torch.load(linear_pth, map_location=torch.device('cpu')))
        model.eval()

        os.makedirs(f"{metric_save_pth}/{tmp_key_test}", exist_ok=True)

        ome_image_pth = glob.glob(f"{args.slide_pth}/*HE*{args.test_key}*.ome.tiff")[0]
        ome_image = pyvips.Image.new_from_file(
            ome_image_pth,
            access="sequential",
        )
        width_dsed = ome_image.width // 32
        height_dsed = ome_image.height // 32
        print(width_dsed, height_dsed)
        wsi_array_pred = np.zeros((len(args.marker_list), width_dsed, height_dsed), dtype=np.uint8)
        wsi_array_gt = np.zeros((len(args.marker_list), width_dsed, height_dsed), dtype=np.uint8)
        print('wsi_array', wsi_array_pred.shape)
        del ome_image
        gc.collect()

        dataset = dataset_pixel_inference(root=args.split_pth, mode='test', args=args,
                                          test_key=args.test_key, marker_list=args.marker_list)
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=0)
        nn = len(trainloader)
        print(len(trainloader))
        ld_t = enumerate(trainloader)
        for i_batch, (image_batch, croods, markers_gt) in ld_t:
            if i_batch % 80 == 0:
                print(f"{i_batch}/{nn}", flush=True)
            image_batch = image_batch.moveaxis(1, -1)

            with torch.inference_mode():
                image_batch = image_batch.cuda()
                outs = model(image_batch)  # b c h w
                for idx, out_tmp in enumerate(outs):
                    out_tmp = F.sigmoid(out_tmp)
                    out_tmp = out_tmp.moveaxis(-1, 1)
                    markers_gt_tmp_marker = markers_gt[idx]
                    # print(out_tmp.shape)
                    # out_tmp_clss = (out_tmp >= 0.5).float()
                    out_tmp_clss = out_tmp

                    for i in range(len(croods)):  # batch
                        tmp_x, tmp_y = copy.deepcopy(croods[i])
                        tmp_x, tmp_y = int(tmp_x) // 32, int(tmp_y) // 32
                        # print(tmp_x, tmp_y)
                        out_tmp_clss_b = out_tmp_clss[i]
                        markers_gt_b = markers_gt_tmp_marker[i]
                        # print('out_tmp_clss_b', out_tmp_clss_b.shape)
                        wsi_array_pred[idx, tmp_x:tmp_x + 4, tmp_y:tmp_y + 4] = (
                                out_tmp_clss_b.cpu().numpy() * 255).astype(np.uint8)
                        wsi_array_gt[idx, tmp_x:tmp_x + 4, tmp_y:tmp_y + 4] = (
                            markers_gt_b.numpy()).astype(np.uint8)

        wsi_array_pred = wsi_array_pred.astype(np.uint8)
        wsi_array_pred = np.moveaxis(wsi_array_pred, 1, -1)
        save_ometiff_from_np(marker_list=marker_list, img=wsi_array_pred,
                             output_path=f"{inference_save_tiff_pth}/{tmp_key_test}_pred.ome.tiff",
                             size=wsi_array_pred.shape)

        wsi_array_gt = wsi_array_gt.astype(np.uint8)
        wsi_array_gt = np.moveaxis(wsi_array_gt, 1, -1)
        save_ometiff_from_np(marker_list=marker_list, img=wsi_array_gt,
                             output_path=f"{inference_save_tiff_pth}/{tmp_key_test}_gt.ome.tiff",
                             size=wsi_array_gt.shape)

        metric_dict = {}
        all_roc_data = []
        for idx1, tmp_marker in enumerate(args.marker_list):
            metric_dict[tmp_marker] = {}
            try:
                print(tmp_marker)
                tmp_cutoff = otsu_cutoff_dict[tmp_marker]

                wsi_array_pred_tmp = wsi_array_pred[idx1]
                wsi_array_gt_tmp = wsi_array_gt[idx1]
                print(wsi_array_pred_tmp.shape, wsi_array_gt_tmp.shape)

                wsi_array_pred_tmp = wsi_array_pred_tmp.flatten()
                wsi_array_gt_tmp = wsi_array_gt_tmp.flatten()
                print(wsi_array_pred_tmp.shape, wsi_array_gt_tmp.shape)

                pred_mask = (wsi_array_pred_tmp > 0)
                wsi_array_pred_tmp = wsi_array_pred_tmp[pred_mask]
                wsi_array_gt_tmp = wsi_array_gt_tmp[pred_mask]
                print(wsi_array_pred_tmp.shape, wsi_array_gt_tmp.shape)

                y_true = (wsi_array_gt_tmp >= tmp_cutoff).astype(np.float64)
                wsi_array_pred_tmp = wsi_array_pred_tmp.astype(np.float64)
                y_pred = wsi_array_pred_tmp / 255

                print(len(y_true), len(y_pred))
                auc = roc_auc_score(y_true, y_pred)
                fpr, tpr, thresholds = roc_curve(y_true, y_pred)

                youden = tpr - fpr
                best_idx = np.argmax(youden)
                best_cutoff = thresholds[best_idx]
                max_youden = youden[best_idx]
                print('best_cutoff', best_cutoff)
                print('max_youden', max_youden)
                y_pred_cls = (y_pred >= best_cutoff).astype(np.float64)

                print(np.unique(y_pred_cls), np.unique(y_true))
                acc = np.mean(y_pred_cls == y_true)

                metric_dict[tmp_marker]["best_cutoff"] = best_cutoff
                metric_dict[tmp_marker]["max_youden"] = max_youden
                metric_dict[tmp_marker]["ACC"] = acc
                metric_dict[tmp_marker]["AUC"] = auc

                print('acc', acc)
                print('auc', auc)

                plt.figure(figsize=(6, 6))
                plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}, ")
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel("False Positive Rate (FPR)")
                plt.ylabel("True Positive Rate (TPR)")
                plt.title("ROC Curve")
                plt.legend()
                # plt.show()
                plt.savefig(f"{metric_save_pth}/{tmp_key_test}/{tmp_marker}.png")
                all_roc_data.append({
                    'marker': tmp_marker,
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': auc
                })
                print()
            except Exception as e:
                print("error", e)
                print()
                continue

        plt.figure(figsize=(8, 8))
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown']
        linestyles = ['-', '--', '-.', ':']

        for i, roc_data in enumerate(all_roc_data):
            marker = roc_data['marker']
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            auc = roc_data['auc']
            if np.isnan(auc):
                continue
            color = colors[i % len(colors)]
            ls = linestyles[i % len(linestyles)]
            plt.plot(fpr, tpr, color=color, lw=2, linestyle=ls,
                     label=f'{marker} (AUC = {auc:.4f})')

        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random Guess')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - All Markers Comparison')
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()
        merge_roc_path = f"{metric_save_pth}/{tmp_key_test}/all_markers_roc.png"
        plt.savefig(merge_roc_path, dpi=300, bbox_inches='tight')
        plt.close()

        df_metrics = pd.DataFrame.from_dict(metric_dict, orient="index")
        df_metrics = df_metrics.reset_index().rename(columns={"index": "marker"})
        df_metrics.to_csv(f"{metric_save_pth}/{tmp_key_test}/metrics_summary.csv", index=False, encoding="utf-8")

        gc.collect()

        print(f"{args.test_key} Done")
