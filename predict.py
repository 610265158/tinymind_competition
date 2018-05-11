import os, urllib

import mxnet as mx



#######predict with augmentation, a suitable N can improve the result
N=10

workdir = os.getcwd()

import argparse
parser = argparse.ArgumentParser(description='Process train and val list.')

parser.add_argument('--data_shape', dest='data_shape',type=int, default=128,  \
                    help='the image shape  (default: 128)')
parser.add_argument('--possibility', dest='possibility',type=bool, default=False,  \
                    help='output the possibility')
parser.add_argument('--network', dest='network',type=str, default='resnext-50',  \
                    help='network used to do predict')
parser.add_argument('--pseduolabel', dest='pseduolabel',type=bool, default=False,  \
                    help='if to pseduolabel')
parser.add_argument('--epoch', dest='epoch',type=int, default=6,  \
                    help='which epoch to used')
parser.add_argument('--tta', dest='tta',type=bool, default=True,  \
                    help='test time augmentation')
parser.add_argument('--set', dest='set',type=str, default='test',  \
                    help='test or train')
args = parser.parse_args()
sym, arg_params, aux_params = mx.model.load_checkpoint('./trained_models/your_model', args.epoch)


mod = mx.mod.Module(symbol=sym, context=mx.gpu())
mod.bind(for_training=False, data_shapes=[('data', (N, 3, args.data_shape, args.data_shape))])
mod.set_params(arg_params, aux_params)

import matplotlib
import csv
matplotlib.rc("savefig", dpi=100)
import cv2
import numpy as np
import pandas as pd
import Myaugmentation
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])

shape_=args.data_shape

eigval = np.array([55.46, 4.794, 1.148])
eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                   [-0.5808, -0.0045, -0.8140],
                   [-0.5836, -0.6948, 0.4203]])
test_augs = [
    mx.image.ForceResizeAug(size=(shape_ + int(0.1 * shape_), shape_ + int(0.1 * shape_))),

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
    if args.tta:
        for f in augs:
            img = f(img)
    return img

def predict(img, mod):
    img_ = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    if img is None:
        return None

    ######batch predict
    if args.tta:
        img_batch=np.zeros(shape=((N, 3, args.data_shape, args.data_shape)))
        for i in range(N):
            img=mx.nd.array(img_.copy())
            img=apply_aug_list(img,test_augs)
            img=img.asnumpy()
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)
            img_batch[i,:,:,:]=img
    else:
        img_batch = np.zeros(shape=((1, 3, args.data_shape, args.data_shape)))
        img = mx.nd.array(img_.copy())
        img = apply_aug_list(img, test_augs)
        img = img.asnumpy()
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)
        img_batch[0, :, :, :] = img
    mod.forward(Batch([mx.nd.array(img_batch)]))

    prob = mod.get_outputs()[0].asnumpy()
    prob = prob.sum(axis=0)/N

    prob = np.squeeze(prob)

    if args.possibility:
        #return the possibility (1,100)
        return prob
    else:
        a = np.argsort(prob)[::-1]

        ##top K=5
        return a[0:5]



csv_file=pd.read_csv('label-test1-fake.csv')

name_list=csv_file['filename']
label=csv_file['label']

######load the label_id
label_file=open('label_syntext.txt','r')
lines = label_file.readlines()
label_dict={}
id_to_label={}
for s in lines:
    label_dict[int(s.split('\t')[1])]=s.split('\t')[0]
    id_to_label[(s.split('\t')[0])] = int(s.split('\t')[1])




###write the Possibility , then an ensemble can be applied, and also a boot
if args.possibility:
    ##output the possibilities
    if args.set=='test':
        csv_writer=csv.writer(open(args.network+'with_p_result.csv','w'))

        for i, singlepic in enumerate(name_list):
            img_path = os.path.join('./test1', singlepic)
            result = predict(img_path, mod)
            tmp_str=result.tolist()
            csv_writer.writerow(tmp_str)
            if i%100==0:
                print(i,' pics')

    if args.set=='train':
        csv_writer = csv.writer(open(args.network + 'with_p-label_result.csv', 'w'))
        head=[]
        head.insert(0,'filename')
        for i in range(100):
            head.append(i)
        head.append('label')
        csv_writer.writerow(head)

        workdir = os.getcwd()
        datadir = os.path.join(workdir, 'train')
        name_list = os.listdir(datadir)

        for name in name_list:
            pics_dir = os.path.join(datadir, name)
            pic_list = os.listdir(pics_dir)

            print('process with ',name)
            for i,img in enumerate(pic_list):
                img_path=os.path.join(pics_dir,img)
                result = predict(img_path, mod)
                tmp_str = result.tolist()
                tmp_str.insert(0,img_path)
                tmp_str.append(str(id_to_label[name]))

                csv_writer.writerow(tmp_str)

                if i % 100 == 0:
                    print(i, ' pics')





###pseduolabel, a Semi supervised learning method,  this scripts will produce pseduolabel.txt
###just concat it with the train.lst and retrain the net (cat pseduolabel.txt train.lst >train.lst),
### generally, you will get a boot
else:
    if args.pseduolabel:
        pseduolabel_list = open('pseduolabel.txt', mode="w+", encoding='utf-8');

    pseduolabel_count=34000
    for i ,singlepic in enumerate(name_list):
        img_path=os.path.join('./test1',singlepic)
        result=predict(img_path, mod)

        tmp_str_predict=(label_dict[result[0]]+label_dict[result[1]]+label_dict[result[2]]+label_dict[result[3]]+label_dict[result[4]])

        if args.pseduolabel:
            tmp_str_pseduolabel=str(pseduolabel_count)+'\t'+str(result[0])+'\t'+workdir+'/test1/'+singlepic+'\n'
            pseduolabel_list.write(tmp_str_pseduolabel)
            pseduolabel_count+=1

        label[i]=tmp_str_predict
        if i%100==0:
            print(i,' pics')

    csv_file.to_csv(args.network+'_result.csv',index=None)

    if args.pseduolabel:
        pseduolabel_list.close()