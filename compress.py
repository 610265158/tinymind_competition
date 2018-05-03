import os, urllib

import mxnet as mx


import argparse
parser = argparse.ArgumentParser(description='Process train and val list.')
parser.add_argument('--img', dest='img',type=str, default='./a.jpg',  \
                    help='the picture to be classified')
parser.add_argument('--epoch', dest='epoch',type=int, default=1,  \
                    help='which model from epoch? default :0')
args = parser.parse_args()

sym, arg_params, aux_params = mx.model.load_checkpoint('./trained_models/your_model', args.epoch)


print(sym.list_arguments())



lzfc1=arg_params['lzfc1_weight']

print(lzfc1.shape)

count=0
for single in lzfc1:
    for ss in single:
        if ss<0.00001:
            count+=1
print(count)




# mod = mx.mod.Module(symbol=sym, context=mx.gpu())
# mod.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
# mod.set_params(arg_params, aux_params)
#
#
#
# import matplotlib
#
# matplotlib.rc("savefig", dpi=100)
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# from collections import namedtuple
#
# Batch = namedtuple('Batch', ['data'])
#
#
#
#
#
# def predict(img, mod):
#     img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
#     if img is None:
#         return None
#     img = cv2.resize(img, (224, 224))
#     img = np.swapaxes(img, 0, 2)
#     img = np.swapaxes(img, 1, 2)
#     img = img[np.newaxis, :]
#
#     mod.forward(Batch([mx.nd.array(img)]))
#     prob = mod.get_outputs()[0].asnumpy()
#     prob = np.squeeze(prob)
#
#     a = np.argsort(prob)[::-1]
#     for i in a[0:2]:
#         print('probability=%f, class=%d' % (prob[i], i))
#
#
# img=args.img
#
# predict(img, mod)