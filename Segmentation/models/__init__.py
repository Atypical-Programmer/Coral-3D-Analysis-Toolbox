from models.deeplab import Deeplab
import torch
import math

nclass=3

def build_model(args):
    if args.model=='deeplabv3p':
        model=Deeplab(backbone=args.backbone,output_stride=16,num_classes=nclass,sync_bn=args.sync_bn,shape_c=True,freeze_bn=False)
    
    return model