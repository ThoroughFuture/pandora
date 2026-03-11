import numpy as np
import torch
from scipy.ndimage import zoom
import torch.nn as nn
import yaml
from argparse import Namespace
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from collections import OrderedDict
import os
from torchvision import transforms
import torch.distributed as dist


def accuracy(outputs, labels):
    correct = (outputs == labels).float()
    acc = correct.sum() / len(correct)
    return acc


def calculate_bacc(y_true, y_pred):
    if not isinstance(y_true, torch.Tensor) or not isinstance(y_pred, torch.Tensor):
        raise TypeError("y_true and y_pred must be torch.Tensor")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be the same shape")
    if not torch.all((y_true == 0) | (y_true == 1)) or not torch.all((y_pred == 0) | (y_pred == 1)):
        raise ValueError("elements must be 0 or 1")
    tp = torch.sum((y_true == 1) & (y_pred == 1))
    tn = torch.sum((y_true == 0) & (y_pred == 0))
    fp = torch.sum((y_true == 0) & (y_pred == 1))
    fn = torch.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn) if (tp + fn) != 0 else torch.tensor(0.0, device=y_true.device)
    tnr = tn / (tn + fp) if (tn + fp) != 0 else torch.tensor(0.0, device=y_true.device)
    bacc = (tpr + tnr) / 2
    return bacc


def ordered_yaml():
    """
	yaml orderedDict support
	"""
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def get_args(args, configs_dir='configs'):
    rank = dist.get_rank()
    if '/configs/' not in args.yml_opt_path:
        args.yml_opt_path = f"./{configs_dir}/{args.yml_opt_path}"
    if '.yml' not in args.yml_opt_path:
        args.yml_opt_path = f"{args.yml_opt_path}.yml"
    if rank == 0:
        print('args.yml_opt_path', args.yml_opt_path, flush=True)

    with open(args.yml_opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        if rank == 0:
            print(f"Loaded configs from {args.yml_opt_path}")
    args2 = Namespace(**config)
    args.__dict__.update(args2.__dict__)

    args.base_lr = float(args.base_lr)
    args.reg = float(args.reg)
    args.bf16 = bool(args.bf16)

    if not hasattr(args, 'train_mode'):
        args.train_mode = 'patch'
    if not hasattr(args, 'use_fp16'):
        args.use_fp16 = False
    args.use_fp16 = bool(args.use_fp16)

    if not hasattr(args, 'in_memory'):
        args.in_memory = False

    os.makedirs(args.save_model_pth, exist_ok=True)

    args.pin_memory = True

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    return args


def get_args_inference(args, configs_dir='configs'):
    if '/configs/' not in args.yml_opt_path:
        args.yml_opt_path = f"./{configs_dir}/{args.yml_opt_path}"
    if '.yml' not in args.yml_opt_path:
        args.yml_opt_path = f"{args.yml_opt_path}.yml"
    print('args.yml_opt_path', args.yml_opt_path, flush=True)

    with open(args.yml_opt_path, mode='r') as f:
        loader, _ = ordered_yaml()
        config = yaml.load(f, loader)
        print(f"Loaded configs from {args.yml_opt_path}")
    args2 = Namespace(**config)
    args.__dict__.update(args2.__dict__)

    args.base_lr = float(args.base_lr)
    args.reg = float(args.reg)
    args.bf16 = bool(args.bf16)

    if not hasattr(args, 'train_mode'):
        args.train_mode = 'patch'
    if not hasattr(args, 'use_fp16'):
        args.use_fp16 = False
    args.use_fp16 = bool(args.use_fp16)

    if not hasattr(args, 'in_memory'):
        args.in_memory = False

    os.makedirs(args.save_model_pth, exist_ok=True)

    args.pin_memory = True

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    return args


def get_transforms(args):
    if args.model_name == "univ2_lora":
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    else:
        raise NotImplementedError
    return transform


import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss


def get_loss(loss_name, args, **kwargs):
    if loss_name == "ce":
        loss = nn.CrossEntropyLoss()
    elif loss_name == "dice":
        loss = DiceLoss(softmax=False, to_onehot_y=True, include_background=True)
    elif loss_name == "mae":
        loss = nn.L1Loss()
    elif loss_name == "mse":
        loss = nn.MSELoss()
    elif loss_name == "bce":
        loss = nn.BCELoss()
    elif loss_name == "bcewl":
        loss = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError
    return loss


def calculate_dice(
        pred: torch.Tensor,
        target: torch.Tensor,
        smooth: float = 1e-6,
        dims: tuple = None
):
    pred = pred.float()
    target = target.float()

    if pred.dim() == target.dim() and pred.size(1) == 1:
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

    if dims is None:
        if pred.dim() == 4:  # [B, C, H, W]
            dims = (1, 2, 3)
        elif pred.dim() == 3:  # [B, H, W]
            dims = (1, 2)
        elif pred.dim() == 2:  # [B,  N]
            dims = (1,)
        else:  # [C, H, W] or [H, W]
            dims = tuple(range(1, pred.dim())) if pred.dim() > 1 else ()

    intersection = torch.sum(pred == target, dim=dims)
    union = pred.flatten().shape[0] * 2
    dice_all = (2.0 * intersection) / (union + smooth)

    intersection = torch.sum(pred * target, dim=dims)
    union = torch.sum(target, dim=dims) * 2
    dice_pos = (2.0 * intersection) / (union + smooth)

    return dice_all, dice_pos
