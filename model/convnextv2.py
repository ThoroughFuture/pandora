# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
#from camel.model.Moe import SparseMoE,SoftMoELayerWrapper,PEER


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x



  
    
class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)

        x = self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        features.append(x)

        return features

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)
        return x



class convnextv2_N(nn.Module):
    def __init__(self, Linear_only=False, num_class=2):
        super().__init__()
        self.convnextv2 = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640])
        
            
        self.convnextv2.head = nn.Sequential()
        self.Linear = nn.Linear(640, num_class)
        self.f = Linear_only
        # feature[0]([2, 80, 64, 64])
        # feature[1]([2, 160, 32, 32])
        # feature[2]([2, 320, 16, 16])
        # feature[3]([2, 640, 8, 8])
        # feature[4]([2, 640])
    def forward(self, x):
        feature = self.convnextv2(x)
        out = feature[-1]

        #out = out.view(x.shape[0], -1)

        out = self.Linear(out)

        if self.f:
            return out
        else:
            return out, feature
        

class convnextv2_T(nn.Module):
    def __init__(self, Linear_only=False, num_class=2):
        super().__init__()
        self.convnextv2 = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        
      

        self.convnextv2.head = nn.Sequential()
        self.Linear = nn.Linear(768, num_class)
        self.f = Linear_only
        # feature[0]([2, 96, 64, 64])
        # feature[1]([2, 192, 32, 32])
        # feature[2]([2, 384, 16, 16])
        # feature[3]([2, 768, 8, 8])
        # feature[4]([2, 768])
    def forward(self, x):
        feature = self.convnextv2(x)
        out = feature[-1]

        #out = out.view(x.shape[0], -1)

        out = self.Linear(out)

        if self.f:
            return out
        else:
            return out, feature



class convnextv2_B(nn.Module):
    def __init__(self, Linear_only=False, num_class=2):
        super().__init__()
        self.convnextv2 = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
       
        self.Linear = nn.Linear(1024, num_class)
        self.f = Linear_only
        # feature[0]([2, 128, 64, 64])
        # feature[1]([2, 256, 32, 32])
        # feature[2]([2, 512, 16, 16])
        # feature[3]([2, 1024, 8, 8])
        # feature[4]([2, 1024])
    def forward(self, x):
        feature = self.convnextv2(x)
        out = feature[-1]

        #out = out.view(x.shape[0], -1)

        out = self.Linear(out)

        if self.f:
            return out
        else:
            return out, feature


class convnextv2_L(nn.Module):
    def __init__(self, Linear_only=False, num_class=2):
        super().__init__()
        self.convnextv2 = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        
       

        self.convnextv2.head = nn.Sequential()
        self.Linear = nn.Linear(1536, num_class)
        self.f = Linear_only

        # feature[0]([2, 192, 64, 64])
        # feature[1]([2, 384, 32, 32])
        # feature[2]([2, 768, 16, 16])
        # feature[3]([2, 1536, 8, 8])
        # feature[4]([2, 1536])
    def forward(self, x):
        feature = self.convnextv2(x)
        out = feature[-1]

        #out = out.view(x.shape[0], -1)
        out = self.Linear(out)

        if self.f:
            return out
        else:
            return out, feature
        
class convnextv2_H(nn.Module):
    def __init__(self, Linear_only=False, num_class=2):
        super().__init__()
        self.convnextv2 = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816])
    
            
        self.Linear = nn.Linear(2816, num_class)
        self.f = Linear_only
        # feature[0]([2, 352, 64, 64])
        # feature[1]([2, 704, 32, 32])
        # feature[2]([2, 1408, 16, 16])
        # feature[3]([2, 2816, 8, 8])
        # feature[4]([2, 2816])
    def forward(self, x):
        feature = self.convnextv2(x)
        out = feature[-1]

        #out = out.view(x.shape[0], -1)

        out = self.Linear(out)

        if self.f:
            return out
        else:
            return out, feature




class convnextv2_L_multi_kd(nn.Module):
    def __init__(self, Linear_only=False):
        super().__init__()
        self.convnextv2 = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
        
        self.convnextv2.head = nn.Sequential()
            

        self.convnextv2.head = nn.Sequential()
        self.Linear1 = nn.Linear(1536, 1024)
        self.Linear2 = nn.Linear(1536, 1280)
        self.Linear3 = nn.Linear(1536, 1536)
        self.Linear4 = nn.Linear(1536, 768)
        self.Linear5 = nn.Linear(1536, 512)
        self.Linear6 = nn.Linear(1536, 1024)
        self.Linear7 = nn.Linear(1536, 1024)
        self.Linear8 = nn.Linear(1536, 1024)
        self.Linear9 = nn.Linear(1536, 2048)
        self.Linear10 = nn.Linear(1536, 1536)
        self.Linear11 = nn.Linear(1536, 768)
      
        self.f = Linear_only


    def forward(self, x):
        feature = self.convnextv2(x)
        out = feature[-1]

        #out = out.view(x.shape[0], -1)
        out1 = self.Linear1(out)
        out2 = self.Linear2(out)
        out3 = self.Linear3(out)
        out4 = self.Linear4(out)
        out5 = self.Linear5(out)
        out6 = self.Linear6(out)
        out7 = self.Linear7(out)
        out8 = self.Linear8(out)
        out9 = self.Linear9(out)
        out10 = self.Linear10(out)
        out11 = self.Linear11(out)
      
        return out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11
  

class convnextv2_B_multi_kd(nn.Module):
    def __init__(self, Linear_only=False):
        super().__init__()
        self.convnextv2 = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])
      
        
        self.convnextv2.head = nn.Sequential()
        self.Linear1 = nn.Linear(1024, 1024)
        self.Linear2 = nn.Linear(1024, 1280)
        self.Linear3 = nn.Linear(1024, 1024)
        self.Linear4 = nn.Linear(1024, 768)
        self.f = Linear_only

        # feature[0]([2, 192, 64, 64])
        # feature[1]([2, 384, 32, 32])
        # feature[2]([2, 768, 16, 16])
        # feature[3]([2, 1024, 8, 8])
        # feature[4]([2, 1024])
    def forward(self, x):
        feature = self.convnextv2(x)
        out = feature[-1]

        #out = out.view(x.shape[0], -1)
        out1 = self.Linear1(out)
        out2 = self.Linear2(out)
        out3 = self.Linear3(out)
        out4 = self.Linear4(out)

        return out1,out2,out3,out4


class convnextv2_N_multi_kd(nn.Module):
    def __init__(self, Linear_only=False):
        super().__init__()
        self.convnextv2 = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640])
        self.convnextv2.head = nn.Sequential()

        self.convnextv2.head = nn.Sequential()
        self.Linear1 = nn.Linear(640, 1024)
        self.Linear2 = nn.Linear(640, 1280)
        self.Linear3 = nn.Linear(640, 1536)
        self.Linear4 = nn.Linear(640, 768)
        self.Linear5 = nn.Linear(640, 512)
        self.Linear6 = nn.Linear(640, 1024)
        self.Linear7 = nn.Linear(640, 1024)
        self.Linear8 = nn.Linear(640, 1024)
        self.Linear9 = nn.Linear(640, 2048)
        self.Linear10 = nn.Linear(640, 1536)
        self.Linear11 = nn.Linear(640, 768)
      
        self.f = Linear_only


    def forward(self, x):
        feature = self.convnextv2(x)
        out = feature[-1]

        #out = out.view(x.shape[0], -1)
        out1 = self.Linear1(out)
        out2 = self.Linear2(out)
        out3 = self.Linear3(out)
        out4 = self.Linear4(out)
        out5 = self.Linear5(out)
        out6 = self.Linear6(out)
        out7 = self.Linear7(out)
        out8 = self.Linear8(out)
        out9 = self.Linear9(out)
        out10 = self.Linear10(out)
        out11 = self.Linear11(out)
      
        return out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11


class convnextv2_T_multi_kd(nn.Module):
    def __init__(self, Linear_only=False):
        super().__init__()
        self.convnextv2 = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
        
        self.convnextv2.head = nn.Sequential()
        self.Linear1 = nn.Linear(768, 1024)
        self.Linear2 = nn.Linear(768, 1280)
        self.Linear3 = nn.Linear(768, 1536)
        self.Linear4 = nn.Linear(768, 768)
        self.Linear5 = nn.Linear(768, 512)
        self.Linear6 = nn.Linear(768, 1024)
        self.Linear7 = nn.Linear(768, 1024)
        self.Linear8 = nn.Linear(768, 1024)
        self.Linear9 = nn.Linear(768, 2048)
        self.Linear10 = nn.Linear(768, 1536)
        self.Linear11 = nn.Linear(768, 768)
      
        self.f = Linear_only


    def forward(self, x):
        feature = self.convnextv2(x)
        out = feature[-1]

        #out = out.view(x.shape[0], -1)
        out1 = self.Linear1(out)
        out2 = self.Linear2(out)
        out3 = self.Linear3(out)
        out4 = self.Linear4(out)
        out5 = self.Linear5(out)
        out6 = self.Linear6(out)
        out7 = self.Linear7(out)
        out8 = self.Linear8(out)
        out9 = self.Linear9(out)
        out10 = self.Linear10(out)
        out11 = self.Linear11(out)
      
        return out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11
