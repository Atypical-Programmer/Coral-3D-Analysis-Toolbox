import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random
from dataloaders.transform import *
import torch.optim as optim
import os

def default_loader(path):
    return cv2.imread(path)
    
class make_dataset(Dataset):
    def __init__(self,txt,loader=default_loader):
        super(make_dataset,self).__init__()
        fh=open(txt,'r')
        slices=[]
        filter_num=0
        for line in fh:
            words=line.strip().split()
            slices,filter_num=self._filter(words,slices,filter_num)
        self.slices=slices
        self.filter_num=filter_num

        self.loader = loader
        self.rand_flip=rand_flip
        self.rand_trans=rand_trans
        self.rand_rotate=rand_rotate
        self.rand_rotate90=rand_rotate90
        print('Filter_num:' + str(self.filter_num) + ' Slices_num:' + str(len(self.slices)))

    def _filter(self,words, slices, filter_num):
        slice_label_path = words[1]
        slice_label = cv2.imread((slice_label_path))
        if np.sum(slice_label) == 0:
            filter_num = filter_num + 1
        else:
            slices.append((words[0], words[1],words[2]))
        return slices, filter_num

    def __getitem__(self, index):
        fn,label,depth=self.slices[index]
        slice_img=self.loader(fn)
        slice_label=self.loader(label)
        slice_depth=self.loader(depth)
        slice_img = cv2.resize(slice_img, (224,224), interpolation=cv2.INTER_AREA)
        slice_label = cv2.resize(slice_label, (224,224), interpolation=cv2.INTER_AREA)  # refinenet 1/4
        slice_depth = cv2.resize(slice_depth, (224,224), interpolation=cv2.INTER_AREA)

        slice_img,slice_label,slice_depth=self.rand_rotate90(slice_img,slice_label,slice_depth)
        slice_img,slice_label,slice_depth=self.rand_flip(slice_img,slice_label,slice_depth)
        slice_img, slice_label,slice_depth = self.rand_trans(slice_img, slice_label,slice_depth)
        slice_img, slice_label,slice_depth = self.rand_rotate(slice_img, slice_label,slice_depth)
   
        slice_label=slice_label[:,:,0]
        c1,c2,c3 = cv2.split(slice_img)
        slice_rgbd = cv2.merge((c1,c2,c3,slice_depth[:,:,0]))
        slice_rgbd=slice_rgbd.transpose([2,0,1])
        slice_img=slice_img.transpose([2,0,1])
        slice_img = torch.from_numpy(slice_img)
        slice_rgbd = torch.from_numpy(slice_rgbd)
        slice_label=torch.from_numpy(slice_label)

        return slice_rgbd,slice_label

    def __len__(self):
        return len(self.slices)

# def label_transform(slice_label):
#         label = torch.from_numpy(slice_label)
#         return label
