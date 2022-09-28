from models.backbone import resnet,xception,resnet_rgbd

def build_backbone(backbone, output_stride, Conv, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, Conv, BatchNorm)
    elif backbone == 'resnet_rgbd':
        return resnet_rgbd.ResNet101(output_stride, Conv, BatchNorm)
    elif  backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    else:
        raise NotImplementedError

