import cv2
import numpy as np
import random

def rand_flip(data,label,depth):
    m=random.randint(0,1)
    angle=random.randint(-1,1)
    if m==0:
       data=cv2.flip(data,angle)
       label=cv2.flip(label,angle)
       depth=cv2.flip(depth,angle)
    return data,label,depth

def rand_trans(data,label,depth):
    pixelx=random.randint(0,50)
    pixely = random.randint(0, 50)
    affine_arr = np.float32([[1, 0, pixelx], [0, 1, pixely]])
    data=cv2.warpAffine(data,affine_arr,(data.shape[0],data.shape[1]))
    label=cv2.warpAffine(label, affine_arr, (label.shape[0], label.shape[1]))
    depth=cv2.warpAffine(depth, affine_arr, (depth.shape[0], depth.shape[1]))
    return data,label,depth

def rand_rotate(data,label,depth):
    degree=random.randint(0,10)
    M = cv2.getRotationMatrix2D((data.shape[0] / 2, data.shape[1] / 2), degree, 1)
    data = cv2.warpAffine(data, M, (data.shape[0], data.shape[1]))
    label= cv2.warpAffine(label, M, (label.shape[0], label.shape[1]))
    depth= cv2.warpAffine(depth, M, (depth.shape[0], depth.shape[1]))
    return data, label,depth

def rand_rotate90(img,label,depth,prob=1.0):
    if random.random() < prob:
        factor = random.randint(0, 4)
        img = np.rot90(img, factor)
        if label is not None:
            label = np.rot90(label, factor)
        if depth is not None:
            depth = np.rot90(depth, factor)
        return img.copy(), label.copy(), depth.copy()