
import os
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
import torch.utils.data.distributed
import sys


from camel.model.convnextv2 import convnextv2_L,convnextv2_H,convnextv2_B,convnextv2_T,convnextv2_T_multi_kd,convnextv2_L_multi_kd,convnextv2_N_multi_kd,convnextv2_B_multi_kd


dist.init_process_group(backend='nccl')
rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)

pt_path = f''
new_ptah = f''

model = convnextv2_L_multi_kd(Linear_only=False).cuda() 
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 转换模型的 BN 层
model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

model.load_state_dict(torch.load(pt_path, map_location=torch.device('cpu')))

new_linear = nn.Linear(1536, 2).cuda()

setattr(model.module, 'Linear', new_linear)


for i in range(1, 5):
    linear_layer_name = f'Linear{i}'
    if hasattr(model.module, linear_layer_name):
        delattr(model.module, linear_layer_name)



torch.save(model.state_dict(), f'{new_ptah}/backbone_moudle_pretrained.pt')


a = torch.load(pt_path, map_location=torch.device('cpu'))  


if isinstance(a, torch.nn.Module):
    raise TypeError("Loaded object is a model, not a state_dict.")
elif isinstance(a, dict):
    state_dict = a
else:
    raise TypeError("Loaded object is not a dict or model.")

new_state_dict = {}

for key, value in state_dict.items():
   
    if key.startswith('module.convnextv2.'):
        key = key[len('module.convnextv2.'):]
    
    if 'Linear' in key:
        continue
    
    new_state_dict[key] = value

torch.save(new_state_dict, f'{new_ptah}/backbone_pretrained.pt')



model = convnextv2_L(Linear_only=False).cuda() 
model = nn.SyncBatchNorm.convert_sync_batchnorm(model)  
model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

model.load_state_dict(torch.load(f'{new_ptah}/backbone_moudle_pretrained.pt', map_location=torch.device('cpu')))

