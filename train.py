# -*- coding:utf-8 -*-
###############
'''
write by lz 2018.5.3, just play with it
'''


############
import mxnet as mx
import random
import os

os.environ['MXNET_CPU_WORKER_NTHREADS'] = '4'

import argparse
parser = argparse.ArgumentParser(description='Start train.')
parser.add_argument('--num_classes', dest='num_classes',type=int, default=100,  \
                    help='the num of the classes (default: 100)')
parser.add_argument('--batch_size', dest='batch_size',type=int, default=128,  \
                    help='batch size (default: 128)')
parser.add_argument('--network', dest='network',type=str, default='resnet-50',  \
                    help='the net structure uesd (default: resnet-50)')
parser.add_argument('--layer_name', dest='layer_name',type=str, default='_plus14,_plus15',  \
                    help='the layers start to train, for finetune, you could play with it (default: concat(_plus14 _plus15))')
parser.add_argument('--viz_net', dest='viz_net',type=bool, default=0,  \
                    help='to viz the net structure or not (default: false)')
parser.add_argument('--epoch', dest='epoch',type=int, default=0,  \
                    help='the models for finetune epoch (default: 0)')
parser.add_argument('--finetune', dest='finetune',type=bool, default=0,  \
                    help='finetune frome a pretrained model (default: false)')
parser.add_argument('--scratch', dest='scratch',type=bool, default=0,  \
                    help='train from sctrach (default: false)')
parser.add_argument('--num_epoch', dest='num_epoch',type=int, default=100,  \
                    help='epoch be trained (default: 100)')
parser.add_argument('--data_shape', dest='data_shape',type=int, default=128,  \
                    help='the image shape  (default: 128)')
parser.add_argument('--log_file', dest='log_file',type=str, default='log.log',  \
                    help='the log file (default: log.log)')
parser.add_argument('--freeze', dest='freeze_layer_pattern', type=str, default="^(stage1|conv0).*",
                    help='freeze layer pattern')
parser.add_argument('--wd', dest='weight_decay', type=float, default=0.005,
                    help='weight decay (default: 0.005),')
parser.add_argument('--num_examples', dest='num_examples', type=int, default=32000,
                    help='num of examples (default: 32000),no use')
args = parser.parse_args()


############play with the parameters below
def get_fine_tune_model(symbol, arg_params, num_classes, layer_name=args.layer_name):

    all_layers = symbol.get_internals()

    layer_names=args.layer_name.split(',')

    #####like the shuffe bet and densenet, just concat different layers, and shuffle them, play with it
    ##### i dont know if the shuffle will help, but i am sure the concat do work in many task
    ##### write by lz 2018.5.3
    #random.shuffle(layer_names)

    layers_embed=[]
    for layer in layer_names:
        net_tmp = all_layers[layer+'_output']
        layers_embed.append(net_tmp)

    net=mx.symbol.concat(*layers_embed)

    net = mx.symbol.BatchNorm(data=net,fix_gamma=False, momentum=0.9, eps=2e-5)
    net = mx.symbol.LeakyReLU(data=net,act_type='prelu')

    embedding=mx.symbol.FullyConnected(data=net,num_hidden=1024)
    embedding = mx.symbol.LeakyReLU(data=embedding, act_type='prelu')
    embedding = mx.symbol.Dropout(data=embedding, p=0.5)
    #embedding = mx.symbol.L2Normalization(data=embedding)
    fc2 = mx.symbol.FullyConnected(data=embedding, num_hidden=num_classes,name='lzfc1')  ####classify layer hidden=num_class
    fc2 = mx.symbol.flatten(data=fc2)
    net = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    new_args = dict({k:arg_params[k] for k in arg_params if layer_name not in k})
    return (net, new_args)


import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)
logger = logging.getLogger()
fh = logging.FileHandler(args.log_file)
logger.addHandler(fh)

def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    import  os
    if not os.access('./trained_models',os.F_OK):
        os.mkdir('./trained_models')
    #lr_scheduler = mx.lr_scheduler.FactorScheduler(80, 0.8)
    lr_scheduler = mx.lr_scheduler.PolyScheduler(4400,0.01, 2)
    epoch_end_callback = mx.callback.do_checkpoint("./trained_models/your_model", 1)

    devs = [mx.gpu(i) for i in range(num_gpus)]
    acc = mx.metric.TopKAccuracy(top_k=5)

    freeze_layer_pattern=args.freeze_layer_pattern

    ###freeze some layers
    import re
    if freeze_layer_pattern.strip():
        re_prog = re.compile(freeze_layer_pattern)
        fixed_param_names = [name for name in symbol.list_arguments() if re_prog.match(name)]
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')


    mod = mx.mod.Module(symbol=symbol,
                        context=devs,
                        data_names=['data'],
                        label_names=['softmax_label'],
                        fixed_param_names=fixed_param_names)

    mod.fit(train, val,
        num_epoch=args.num_epoch,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        epoch_end_callback=epoch_end_callback,
        kvstore='device',
        optimizer='NAG',
        optimizer_params={
            'learning_rate':0.01,
            'momentum': 0.9,
            'lr_scheduler': lr_scheduler,
            'wd':args.weight_decay
        },
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        eval_metric=acc)

import  Mydataiter
train_iter, val_iter=Mydataiter.get_iterator(args.batch_size,args.data_shape)

if args.finetune :
    print('finetune from pretrained model ...with', args.network)

    sym, arg_params, aux_params = mx.model.load_checkpoint('./models/' + args.network, args.epoch)
    (new_sym, new_args) = get_fine_tune_model(sym, arg_params, args.num_classes)

    if args.viz_net:
        #print(new_sym.list_arguments())
        b = mx.viz.plot_network(new_sym)#可视化网络结构
        b.view()

    fit(new_sym, new_args, aux_params, train_iter, val_iter, args.batch_size,num_gpus=1)

if args.scratch:

    print('train from scratch...with',args.network)
    if args.network=='mobilenet':
        from symbols import  mobilenet
        sym=mobilenet.get_symbol(args.num_classes)

    elif args.network=='mobilenet_v2':
        from symbols import mobilenetv2
        sym=mobilenetv2.get_symbol(args.num_classes)

    elif args.network=='resnext':
        from symbols import resnext

        ###you should set the params to get the symbol
        #sym=resnext.get_symbol(***)
    elif args.network=='inception-resnet-v2':
        from symbols import inception_resnet_v2
        sym=inception_resnet_v2.get_symbol(num_classes=args.num_classes)
    else:
        ###implement your symbol
        raise Exception("You should gives a symbol")

    if args.viz_net:
        print(new_sym.list_arguments())
        b = mx.viz.plot_network(new_sym)#可视化网络结构
        b.view()

    lr_scheduler = mx.lr_scheduler.FactorScheduler(40, 0.8)
    epoch_end_callback = mx.callback.do_checkpoint("./trained_models/your_model", 1)

    devs = mx.gpu()

    mod = mx.mod.Module(symbol=sym,
                        context=devs,
                        data_names=['data'],
                        label_names=['softmax_label'])


    acc = mx.metric.TopKAccuracy(top_k=5)


    mod.fit(train_iter, val_iter,
            num_epoch=args.num_epoch,
            allow_missing=True,
            batch_end_callback=mx.callback.Speedometer(args.batch_size, 5),
            epoch_end_callback=epoch_end_callback,
            kvstore='device',
            optimizer='NAG',
            optimizer_params={
                'learning_rate': 0.04,
                'momentum': 0.9,
                'lr_scheduler': lr_scheduler,
                'wd': 0.005
            },
            initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
            eval_metric=acc)


























