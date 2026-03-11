import argparse
import os
import random
import numpy as np
import torch
from utils.trainer_ddp_multi_marker import trainer_ddp
from utils.utils import get_args
import torch.distributed as dist
import shutil
from models.linear_head import linear_head
from utils.label_mapping import all_marker_tuple
import warnings

warnings.filterwarnings('ignore')

def run_ddp():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int,
                        help='node rank for distributed training')
    parser.add_argument('--yml_opt_path', type=str, default='linear_768_marker_all')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    args = get_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.marker_list == "all":
        args.marker_list = all_marker_tuple

    if rank == 0:
        print('args', args, flush=True)
        print('len args.marker_list', len(args.marker_list))

    args.is_pretrain = True
    args.exp = 'TU'
    snapshot_path = f"{args.snapshot_path}/{args.exp}"
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    snapshot_path = snapshot_path + '_testkey' + str(args.test_key)

    args.save_model_pth = f"{args.save_model_pth}/{snapshot_path.split(os.sep)[-1]}"

    if rank == 0:
        print('args.save_model_pth', args.save_model_pth, flush=True)
    if rank == 0:
        print('snapshot_path', snapshot_path, flush=True)
    os.makedirs(args.save_model_pth, exist_ok=True)

    shutil.rmtree(snapshot_path, ignore_errors=True)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path, exist_ok=True)

    model = linear_head(marker_num=len(args.marker_list), in_dim=args.in_dim)
    trainer_ddp(args, model, snapshot_path)

    if dist.is_initialized():
        dist.destroy_process_group()

    if rank == 0:
        print('Done', flush=True)

if __name__ == "__main__":
    run_ddp()
