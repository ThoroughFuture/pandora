from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import argparse
import logging
import os
import random
import sys
import time
import numpy as np
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

def gather_tensor(tensor, dim=0):
    """
    Gather tensor from all processes and concatenate on `dim`.
    Returns: concatenated tensor on all ranks (same result on all ranks).
    """
    world_size = dist.get_world_size()
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=dim)


def trainer_ddp(args, model, snapshot_path):
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()

    if dist.get_rank() == 0:
        if args.pre_trained:
            print(f"{args.save_model_pth}/last.pth", os.path.exists(f"{args.save_model_pth}/last.pth"))
            if os.path.exists(f"{args.save_model_pth}/last.pth"):
                try:
                    pretrained_path = f"{args.save_model_pth}/last.pth"
                    pretrained_weights = torch.load(pretrained_path, map_location='cpu')
                    model.load_state_dict(pretrained_weights, strict=True)
                    print('load model from:', pretrained_path, flush=True)
                except Exception as e:
                    print(f"Error loading pretrained model: {e}", flush=True)

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    if rank == 0:
        logging.info(str(args))
    base_lr = args.base_lr

    batch_size_per_gpu = args.batch_size_per_gpu
    train_dataset = marker_dataset_pixel(root=args.split_pth, mode=args.mode, args=args, test_key=args.test_key)
    if dist.get_rank() == 0:
        print("The length of train set is: {}".format(len(train_dataset)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    worker_init_fn(rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size_per_gpu, num_workers=args.num_workers,
                                              persistent_workers=True, pin_memory=True,
                                              sampler=train_sampler)

    dist.barrier()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model.train()
    # model = nn.DataParallel(model, device_ids=gpu_ids)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                find_unused_parameters=True)
    model = model.cuda()

    loss_t = get_loss(loss_name="bcewl", args=args)
    if rank == 0:
        print('loss_t', loss_t)

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=base_lr, weight_decay=args.reg)

    scheduler = CosineAnnealingLR(optimizer, T_max=6000, eta_min=0.000001)

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    if rank == 0:
        logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    scaler = GradScaler()

    max_iter = args.max_iter
    # iter_view = len(trainloader) // 10
    # iter_view = 1
    iter_view = max(40, len(trainloader) // 10)
    iter_save = iter_view * 5
    if dist.get_rank() == 0:
        print("iter_view", iter_view, "iter_save", iter_save)
        print('args.bf16', args.bf16, flush=True)
    if args.tqdm:
        epoch_iterator = tqdm(range(max_epoch), ncols=70)
    else:
        epoch_iterator = range(max_epoch)
    for epoch_num in epoch_iterator:
        train_sampler.set_epoch(epoch_num)
        # if rank == 0:
        #     logging.info(f'epoch {epoch_num + 1}/{max_epoch}, iter_num {iter_num}/{max_iterations}')
        st = time.time()
        for _, (image_batch, label_batch) in enumerate(trainloader):
            # try:
            optimizer.zero_grad()
            with autocast(enabled=args.bf16):
                loss = 0
                acc_list = []
                bacc_list = []
                for idx, label_t in enumerate(label_batch):
                    image_in = image_batch[idx].cuda()
                    outs = model(image_in)  # b c h w
                    out_tmp = outs[idx]
                    label_t = label_t.cuda()
                    label_t = label_t.unsqueeze(1).to(out_tmp.dtype)
                    loss_tmp = loss_t(out_tmp, label_t)
                    loss += loss_tmp

                    if iter_num % iter_view == 0:
                        out_t = F.sigmoid(out_tmp)
                        out_t = (out_t >= 0.5).float()

                        acc_t = accuracy(out_t, label_t)
                        bacc_t = calculate_bacc(out_t, label_t)
                        acc_list.append(acc_t.item())
                        bacc_list.append(bacc_t.item())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_ = optimizer.param_groups[0]['lr']
            scheduler.step()

            if rank == 0:
                if iter_num % iter_view == 0:
                    ed = time.time()
                    tm = ed - st
                    st = time.time()

                    writer.add_scalar('info/lr', lr_, iter_num)
                    writer.add_scalar('info/total_loss', loss, iter_num)
                    logging.info(
                        f'iteration {iter_num:d}, loss: {loss.item():.4f}, '
                        f'acc_avg: {sum(acc_list) / len(acc_list):.4f}, '
                        f'bacc_avg: {sum(bacc_list) / len(bacc_list):.4f}, '
                        f'lr: {lr_:.6f}, cost time: {tm:.2f}s')

                    print(flush=True)

                if iter_num % iter_save == 0:
                    save_mode_path = f"{args.save_model_pth}/iter_{str(iter_num)}.pth"
                    torch.save(model.module.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                    torch.save(model.module.state_dict(), f"{args.save_model_pth}/last.pth")
                    logging.info(f"save model to {args.save_model_pth}/last.pth")
                    print(flush=True)

            iter_num = iter_num + 1

        gc.collect()

        if epoch_num % args.save_model_freq_epoch == 0:
            if rank == 0:
                save_mode_path = f"{args.save_model_pth}/epoch_{str(epoch_num)}.pth"
                torch.save(model.module.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                torch.save(model.module.state_dict(), f"{args.save_model_pth}/last.pth")
                logging.info(f"save model to {args.save_model_pth}/last.pth")
                print(flush=True)

        if epoch_num >= max_epoch:
            if rank == 0:
                save_mode_path = f"{args.save_model_pth}/epoch_{str(epoch_num)}.pth"
                torch.save(model.module.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                torch.save(model.module.state_dict(), f"{args.save_model_pth}/last.pth")
                logging.info(f"save model to {args.save_model_pth}/last.pth")
                print('break')
            break

    writer.close()
    return "Training Finished!"
