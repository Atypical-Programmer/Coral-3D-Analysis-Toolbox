from turtle import back
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.shape_conv import ShapeConv2d
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.backbone import build_backbone
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class Deeplab(nn.Module):
    def __init__(self,backbone='resnet',output_stride=16,num_classes=3,sync_bn=False,shape_c=False,freeze_bn=False):
        super(Deeplab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if shape_c == True:
            Conv = ShapeConv2d
        else:
            Conv = nn.Conv2d

        self.backbone = build_backbone(backbone, output_stride, Conv, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        # self.attention=build_attention(backbone)

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()