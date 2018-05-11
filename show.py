import os, urllib

import mxnet as mx



#######predict with augmentation, a suitable N can improve the result
N=9

workdir = os.getcwd()



import matplotlib.pyplot as plt
import csv
import cv2
import numpy as np
import pandas as pd
import Myaugmentation
import random
from collections import namedtuple





shape_=128

eigval = np.array([55.46, 4.794, 1.148])
eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                   [-0.5808, -0.0045, -0.8140],
                   [-0.5836, -0.6948, 0.4203]])
test_augs = [
    mx.image.ForceResizeAug(size=(shape_ + int(0.1 * shape_), shape_ + int(0.2 * shape_))),

    mx.image.RandomCropAug((shape_, shape_)),
    ##flip not suitable for charactor
    # mx.image.HorizontalFlipAug(0.5),
    mx.image.CastAug(),
    Myaugmentation.BlurAug(0.5, (5, 5)),
    mx.image.ColorJitterAug(0.1, 0.1, 0.1),
    mx.image.HueJitterAug(0.5),
    mx.image.LightingAug(0.5, eigval, eigvec),
    mx.image.RandomGrayAug(0.5),
    # #### extra augmentation
    Myaugmentation.RandomRotateAug(10, 0.5),

    Myaugmentation.BlurAug(0.5, (7, 7)),

    Myaugmentation.Castint8Aug(0.3)

]

def apply_aug_list(img, augs):

    for f in augs:
        img = f(img)
    return img

def show(img_paths):

    img_batch = np.zeros(shape=((3*(shape_), 3*(shape_),3)))
    z=0
    for i in range(3):
        for j in range(3):
            path = os.path.join('./test1', img_paths[z])
            img_ = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            if img_ is None:
                return None

            ######batch predict

            img=mx.nd.array(img_.copy())
            img=apply_aug_list(img,test_augs)
            img=img.asnumpy()
            img=img/255
            img_batch[i*shape_:(i+1)*shape_,j*shape_:(j+1)*shape_,:]=img

            z+=1

    #plt.imshow(img_batch)
    #plt.show()
    cv2.imshow('res',img_batch)
    cv2.waitKey()



csv_file=pd.read_csv('label-test1-fake.csv')

name_list=csv_file['filename'].tolist()





index_tmp=random.sample(range(1,10000), N)
img_paths=[]
for i in index_tmp:
    img_paths.append(name_list[i])
show(img_paths)









feature=0
if feature:

    print('teset set pic features, ')
    size_feature ={}
    shape=np.asarray([0,0,0])
    for img in name_list:
        img_path=os.path.join('./test1', img)
        im = cv2.imread(img_path)

        shape+=im.shape
        #size_feature[shape]=size_feature[shape]+1 if shape in size_feature else 1

    ##for feature,num in size_feature.items():
        #print('size ',feature," :",num)

    print("test set average size is ",shape/10000)









