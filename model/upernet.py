import warnings
import torch.nn.functional as F
import torch
import torch.nn as nn
from camel.model.camel_feature import Convnextv2_L_feature,Convnextv2_H_feature,Convnextv2_B_feature,Hiera_T_feature,Convnextv2_N_feature,Convnextv2_T_feature

warnings.simplefilter("ignore")




class ConvBnAct(nn.Module):

    def __init__(self, in_channel, out_channel, kernel, stride, padding, dilation=1, bias=False, act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel, stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super(PyramidPoolingModule, self).__init__()
        inter_channels = in_channels // 4
        self.cba1 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba2 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba3 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.cba4 = ConvBnAct(in_channels, inter_channels, 1, 1, 0)
        self.out = ConvBnAct(in_channels * 2, out_channels, 1, 1, 0)

    def pool(self, x, size):
        return nn.AdaptiveAvgPool2d(size)(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode="bilinear", align_corners=True)

    def forward(self, x):
        size = x.shape[2:]
        f1 = self.upsample(self.cba1(self.pool(x, 1)), size)
        f2 = self.upsample(self.cba2(self.pool(x, 2)), size)
        f3 = self.upsample(self.cba3(self.pool(x, 3)), size)
        f4 = self.upsample(self.cba4(self.pool(x, 6)), size)
        f = torch.cat([x, f1, f2, f3, f4], dim=1)
        return self.out(f)



class FeaturePyramidNet(nn.Module):

    def __init__(self, fpn_dim=256,size=[128,256,512],dim=[1024,512,256],feature_name=['feature4','feature3','feature2','feature1'],down_size=512):
        self.fpn_dim = fpn_dim
        super(FeaturePyramidNet, self).__init__()
        self.size = size
        self.down_size = down_size
        self.extra = bool(len(feature_name)>4)
        self.feature_name = feature_name

        self.cba3 = ConvBnAct(dim[1], self.fpn_dim, 1, 1, 0)
        self.cba2 = ConvBnAct(dim[2], self.fpn_dim, 1, 1, 0)
        self.cba1 = ConvBnAct(dim[3], self.fpn_dim, 1, 1, 0)

        self.cba_out3 = ConvBnAct(self.fpn_dim, self.fpn_dim, 1, 1, 0)
        self.cba_out2 = ConvBnAct(self.fpn_dim, self.fpn_dim, 1, 1, 0)
        self.cba_out1 = ConvBnAct(self.fpn_dim, self.fpn_dim, 1, 1, 0)

        if self.extra:
            self.cba0 = ConvBnAct(dim[4], self.fpn_dim, 1, 1, 0)
            self.cba_out0 = ConvBnAct(self.fpn_dim, self.fpn_dim, 1, 1, 0)


    def forward(self, feature):
 
        f4 = feature[self.feature_name[0]]
        f_out4= f4
        f4 = F.interpolate(f4, ( self.size[1], self.size[1]), mode='bilinear', align_corners=False)

        f3 = self.cba3(feature[self.feature_name[1]])+f4
        f_out3 = self.cba_out3(f3)
        f3 = F.interpolate(f3, ( self.size[2], self.size[2]), mode='bilinear', align_corners=False)
        
        f2 = self.cba2(feature[self.feature_name[2]]) + f3
        f_out2 = self.cba_out2(f2)
        f2 = F.interpolate(f2, ( self.size[3], self.size[3]), mode='bilinear', align_corners=False)
        f1= self.cba1(feature[self.feature_name[3]]) + f2
        f_out1 = self.cba_out1(f1)

        

        f_out1 = F.interpolate(f_out1, (self.down_size, self.down_size), mode='bilinear', align_corners=False)
        f_out2 = F.interpolate(f_out2, (self.down_size, self.down_size), mode='bilinear', align_corners=False)
        f_out3 = F.interpolate(f_out3, (self.down_size, self.down_size), mode='bilinear', align_corners=False)
        f_out4= F.interpolate(f_out4, (self.down_size, self.down_size), mode='bilinear', align_corners=False)
        


        if self.extra:
            f1 = F.interpolate(f1, ( self.size[4], self.size[4]), mode='bilinear', align_corners=False)

            f0= self.cba0(feature[self.feature_name[4]]) + f1
            
            f_out0 = self.cba_out0(f0)
            f_out0= F.interpolate(f_out0, (self.down_size, self.down_size), mode='bilinear', align_corners=False)
            out = torch.cat([f_out0,f_out1,f_out2,f_out3,f_out4], dim=1)

            return out
    
        else:
            out = torch.cat([f_out1,f_out2,f_out3,f_out4], dim=1)
            return out
            

 

    

    

class UPerNet_Convnextv2_L(nn.Module):

    def __init__(self,  fpn_dim=256,out_dim=1,output_size=2048):
        super(UPerNet_Convnextv2_L, self).__init__()
        self.backbone = Convnextv2_L_feature(input_size =output_size )
        self.fpn_dim = fpn_dim
        self.output_size = output_size

        self.ppm = PyramidPoolingModule(1536, self.fpn_dim)
        self.fpn = FeaturePyramidNet(self.fpn_dim,size=[output_size//32,output_size//16,output_size//8,output_size//4],dim=[1536,768,384,192],feature_name=['feature4','feature3','feature2','feature1'],down_size=self.output_size//4)
        self.fuse = ConvBnAct(fpn_dim * 4, fpn_dim//4, 3, 1, 1)  # 64
        
        self.out = nn.Conv2d(fpn_dim//4, out_dim, 1, 1, 0)

    def forward(self, x):
        #with torch.no_grad():
        feature = self.backbone(x)
        ppm = self.ppm(feature['feature4'])
        feature.update({'feature4': ppm})
        fpn = self.fpn(feature)
        out = self.fuse(fpn)

        out = self.out(F.interpolate(out, (self.output_size,self.output_size), mode='bilinear', align_corners=False))

        return out


class UPerNet_Convnextv2_H(nn.Module):

    def __init__(self,  fpn_dim=352,out_dim=1,output_size=2048):
        super(UPerNet_Convnextv2_H, self).__init__()
        self.backbone = Convnextv2_H_feature(input_size =output_size )
        self.fpn_dim = fpn_dim
        self.output_size = output_size
        self.ppm = PyramidPoolingModule(2816, self.fpn_dim)
        self.fpn = FeaturePyramidNet(self.fpn_dim,size=[output_size//32,output_size//16,output_size//8,output_size//4],dim=[2816,1408,704,352],feature_name=['feature4','feature3','feature2','feature1'],down_size=self.output_size//4)
        self.fuse = ConvBnAct(fpn_dim * 4, fpn_dim//4, 3, 1, 1)  # 64

        self.out = nn.Conv2d(fpn_dim//4, out_dim, 1, 1, 0)

    def forward(self, x):
        #with torch.no_grad():
        feature = self.backbone(x)
        ppm = self.ppm(feature['feature4'])
        feature.update({'feature4': ppm})
        fpn = self.fpn(feature)
        out = self.fuse(fpn)

        out = self.out(F.interpolate(out, (self.output_size,self.output_size), mode='bilinear', align_corners=False))

        return out



class UPerNet_Convnextv2_B(nn.Module):

    def __init__(self,  fpn_dim=384,out_dim=1,output_size=2048):
        super(UPerNet_Convnextv2_B, self).__init__()
        self.backbone = Convnextv2_B_feature()
        self.fpn_dim = fpn_dim
        self.output_size = output_size

        self.ppm = PyramidPoolingModule(1024, self.fpn_dim)
        self.fpn = FeaturePyramidNet(self.fpn_dim,size=[output_size//32,output_size//16,output_size//8,output_size//4],dim=[1024,512,256,128],feature_name=['feature4','feature3','feature2','feature1'],down_size=self.output_size//4)
        self.fuse = ConvBnAct(fpn_dim * 4, fpn_dim//4, 3, 1, 1)  # 64

        self.out = nn.Conv2d(fpn_dim//4, out_dim, 1, 1, 0)

    def forward(self, x):
        #with torch.no_grad():
        feature = self.backbone(x)
        ppm = self.ppm(feature['feature4'])
        feature.update({'feature4': ppm})
        fpn = self.fpn(feature)
        out = self.fuse(fpn)

        out = self.out(F.interpolate(out, (self.output_size,self.output_size), mode='bilinear', align_corners=False))

        return out


class UPerNet_Convnextv2_T(nn.Module):

    def __init__(self,  fpn_dim=384,out_dim=1,output_size=2048):
        super(UPerNet_Convnextv2_T, self).__init__()
        self.backbone = Convnextv2_T_feature()
        self.fpn_dim = fpn_dim
        self.output_size = output_size

        self.ppm = PyramidPoolingModule(768, self.fpn_dim)
        self.fpn = FeaturePyramidNet(self.fpn_dim,size=[output_size//32,output_size//16,output_size//8,output_size//4],dim=[768,384,192,96],feature_name=['feature4','feature3','feature2','feature1'],down_size=self.output_size//4)
        self.fuse = ConvBnAct(fpn_dim * 4, fpn_dim//4, 3, 1, 1)  # 64

        self.out = nn.Conv2d(fpn_dim//4, out_dim, 1, 1, 0)

    def forward(self, x):
        #with torch.no_grad():
        with torch.no_grad():
            feature = self.backbone(x)
        ppm = self.ppm(feature['feature4'])
        feature.update({'feature4': ppm})
        fpn = self.fpn(feature)
        out = self.fuse(fpn)

        out = self.out(F.interpolate(out, (self.output_size,self.output_size), mode='bilinear', align_corners=False))

        return out
    


class UPerNet_Convnextv2_N(nn.Module):

    def __init__(self,  fpn_dim=512,out_dim=1,output_size=2048):
        super(UPerNet_Convnextv2_N, self).__init__()
        self.backbone = Convnextv2_N_feature()
        self.fpn_dim = fpn_dim
        self.output_size = output_size

        self.ppm = PyramidPoolingModule(640, self.fpn_dim)
        self.fpn = FeaturePyramidNet(self.fpn_dim,size=[output_size//32,output_size//16,output_size//8,output_size//4],dim=[640,320,160,80],feature_name=['feature4','feature3','feature2','feature1'],down_size=self.output_size//4)
        self.fuse = ConvBnAct(fpn_dim * 4, fpn_dim//4, 3, 1, 1)  # 64

        self.out = nn.Conv2d(fpn_dim//4, out_dim, 1, 1, 0)

    def forward(self, x):
        #with torch.no_grad():
        feature = self.backbone(x)
        ppm = self.ppm(feature['feature4'])
        feature.update({'feature4': ppm})
        fpn = self.fpn(feature)
        out = self.fuse(fpn)

        out = self.out(F.interpolate(out, (self.output_size,self.output_size), mode='bilinear', align_corners=False))

        return out
    
