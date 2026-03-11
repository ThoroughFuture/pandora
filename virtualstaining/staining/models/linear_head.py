import copy

import torch
import numpy as np
from sklearn.cluster import KMeans
from torchvision import transforms
from torch import nn


class FFN_head_sigmoid(nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Sequential(*[
            nn.Linear(in_dim, in_dim // 2),
            nn.LayerNorm(in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, out_dim),
            # nn.Sigmoid()
        ])

    def forward(self, x):
        out = self.linear(x)
        return out


class linear_head(nn.Module):
    def __init__(self, marker_num=9, in_dim=768):
        super().__init__()
        self.linear_head = nn.Sequential(*[FFN_head_sigmoid(in_dim, 1) for _ in range(marker_num)])

    def forward(self, feature):
        # convnextv2 in 256 -> feature list
        # torch.Size([2, 96, 64, 64])
        # torch.Size([2, 192, 32, 32])
        # torch.Size([2, 384, 16, 16])
        # torch.Size([2, 768, 8, 8])
        # torch.Size([2, 768])
        out_fin = []
        for idx, head in enumerate(self.linear_head):
            in_feature = feature
            tmp_out = head(in_feature)
            out_fin.append(tmp_out)

        return out_fin


if __name__ == "__main__":
    model = linear_head()

    in_img = torch.rand((2, 384))
    print(in_img.shape)

    feature, out = model(in_img)
    # print(out)

    for x in out:
        print(x.shape)
