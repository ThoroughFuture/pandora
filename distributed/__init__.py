import torch.distributed as dist

import os
import torch

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()



def get_ddp_generator(parament):
    seed = parament.seed
    local_rank = int(os.environ['LOCAL_RANK'])
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


def distributed_concat(tensor, num_total_examples):
    
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    
    return concat[:num_total_examples]


def distributed_concat_cpu(tensor, num_total_examples):
   
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    
    return concat[:num_total_examples].detach().cpu()