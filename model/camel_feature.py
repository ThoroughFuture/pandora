
import torch
import torch.nn as nn
from camel.utils import slice_image,merge_image
from camel.model.convnextv2 import convnextv2_L,convnextv2_H,convnextv2_B,convnextv2_T,convnextv2_N
import os
import torch.distributed as dist


class Convnextv2_L_feature(nn.Module):
    def __init__(self,input_size =2048):
        super().__init__()
        self.Convnext = convnextv2_L().cuda()
        rank = int(os.environ['LOCAL_RANK'])
        self.Convnext = nn.parallel.DistributedDataParallel(self.Convnext, device_ids=[rank], find_unused_parameters=True)

        self.Convnext.Linear = nn.Sequential()

        self.input_size = input_size
        # feature[0]([2, 192, 64, 64])
        # feature[1]([2, 384, 32, 32])
        # feature[2]([2, 768, 16, 16])
        # feature[3]([2, 1536, 8, 8])
        # feature[4]([2, 1536])

    def forward(self, x):
        # x b 3 2048 2048
        x = slice_image(x)
        # b*64  3  256 256
        out = self.Convnext(x)[-1]

        f1 = merge_image(out[0], input_size=64, orginal_size=self.input_size//4)
        f2 = merge_image(out[1], input_size=32, orginal_size=self.input_size//8)
        f3 = merge_image(out[2], input_size=16, orginal_size=self.input_size//16)
        f4 = merge_image(out[3], input_size=8, orginal_size=self.input_size//32)

        feature = {'feature1': f1, 'feature2': f2, 'feature3': f3, 'feature4': f4}
        # feature1 B*192*512*512
        # feature2 B*384*256*256
        # feature3 B*768*128*128
        # feature4 B*1536*64*64
 

        return feature
    


class Convnextv2_H_feature(nn.Module):
    def __init__(self,input_size =2048):
        super().__init__()
        self.Convnext = convnextv2_H().cuda()
        rank = int(os.environ['LOCAL_RANK'])
        self.Convnext = nn.parallel.DistributedDataParallel(self.Convnext, device_ids=[rank], find_unused_parameters=True)

        self.Convnext.Linear = nn.Sequential()
        self.input_size = input_size
        # feature[0]([2, 352, 64, 64])
        # feature[1]([2, 704, 32, 32])
        # feature[2]([2, 1408, 16, 16])
        # feature[3]([2, 2816, 8, 8])
        # feature[4]([2, 2816])

    def forward(self, x):
        x = slice_image(x)
        
        out = self.Convnext(x)[-1]

        f1 = merge_image(out[0], input_size=64, orginal_size=self.input_size//4)
        f2 = merge_image(out[1], input_size=32,orginal_size=self.input_size//8)
        f3 = merge_image(out[2], input_size=16, orginal_size=self.input_size//16)
        f4 = merge_image(out[3], input_size=8, orginal_size=self.input_size//32)

        feature = {'feature1': f1, 'feature2': f2, 'feature3': f3, 'feature4': f4}
        # feature1 B*352*512*512
        # feature2 B*704*256*256
        # feature3 B*1408*128*128
        # feature4 B*2816*64*64


        return feature


class Convnextv2_N_feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.Convnext = convnextv2_N().cuda()
        rank = int(os.environ['LOCAL_RANK'])
        self.Convnext = nn.parallel.DistributedDataParallel(self.Convnext, device_ids=[rank], find_unused_parameters=True)

        self.Convnext.Linear = nn.Sequential()
         # feature[0]([2, 80, 64, 64])
        # feature[1]([2, 160, 32, 32])
        # feature[2]([2, 320, 16, 16])
        # feature[3]([2, 640, 8, 8])
        # feature[4]([2, 640])

    def forward(self, x):
        x = slice_image(x)
        
        out = self.Convnext(x)[-1]

        f1 = merge_image(out[0], input_size=64, orginal_size=512)
        f2 = merge_image(out[1], input_size=32, orginal_size=256)
        f3 = merge_image(out[2], input_size=16, orginal_size=128)
        f4 = merge_image(out[3], input_size=8, orginal_size=64)

        feature = {'feature1': f1, 'feature2': f2, 'feature3': f3, 'feature4': f4}

 
        return feature
    


class Convnextv2_T_feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.Convnext = convnextv2_T().cuda()
        rank = int(os.environ['LOCAL_RANK'])
        self.Convnext = nn.parallel.DistributedDataParallel(self.Convnext, device_ids=[rank], find_unused_parameters=True)
        self.Convnext.Linear = nn.Sequential()
         # feature[0]([2, 96, 64, 64])
        # feature[1]([2, 192, 32, 32])
        # feature[2]([2, 384, 16, 16])
        # feature[3]([2, 768, 8, 8])
        # feature[4]([2, 768])

    def forward(self, x):
        x = slice_image(x)
        
        out = self.Convnext(x)[-1]

        f1 = merge_image(out[0], input_size=64, orginal_size=512)
        f2 = merge_image(out[1], input_size=32, orginal_size=256)
        f3 = merge_image(out[2], input_size=16, orginal_size=128)
        f4 = merge_image(out[3], input_size=8, orginal_size=64)

        feature = {'feature1': f1, 'feature2': f2, 'feature3': f3, 'feature4': f4}

 
        return feature
    

class Convnextv2_B_feature(nn.Module):
    def __init__(self):
        super().__init__()
        self.Convnext = convnextv2_B().cuda()
        rank = int(os.environ['LOCAL_RANK'])
        self.Convnext = nn.parallel.DistributedDataParallel(self.Convnext, device_ids=[rank], find_unused_parameters=True)

        self.Convnext.Linear = nn.Sequential()
        # feature[0]([2, 128, 64, 64])
        # feature[1]([2, 256, 32, 32])
        # feature[2]([2, 512, 16, 16])
        # feature[3]([2, 1024, 8, 8])

    def forward(self, x):
        x = slice_image(x)
        
        out = self.Convnext(x)[-1]

        f1 = merge_image(out[0], input_size=64, orginal_size=512)
        f2 = merge_image(out[1], input_size=32, orginal_size=256)
        f3 = merge_image(out[2], input_size=16, orginal_size=128)
        f4 = merge_image(out[3], input_size=8, orginal_size=64)

        feature = {'feature1': f1, 'feature2': f2, 'feature3': f3, 'feature4': f4}
        # feature1 B*128*512*512
        # feature2 B*256*256*256
        # feature3 B*512*128*128
        # feature4 B*1024*64*64
 
        return feature
    
    
