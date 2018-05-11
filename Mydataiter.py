
import mxnet as mx
import os

class custom_iter(mx.io.DataIter):
    def __init__(self, data_iter):
        super(custom_iter,self).__init__()
        self.data_iter = data_iter
        self.batch_size = self.data_iter.batch_size

    @property
    def provide_data(self):
        return self.data_iter.provide_data

    @property
    def provide_label(self):
        provide_label = self.data_iter.provide_label[0]

        #return [('softmax_label', provide_label[1]), \
        #         ('center_loss_label', provide_label[1])]

        return [('softmax_label', provide_label[1])]

    def hard_reset(self):
        self.data_iter.hard_reset()

    def reset(self):
        self.data_iter.reset()

    def next(self):
        batch = self.data_iter.next()
        label = batch.label[0]

        return mx.io.DataBatch(data=batch.data, label=[label], \
                pad=batch.pad, index=batch.index)


def get_iterator(batch_size,data_shape):
    """return train and val iterators for mnist"""
    # download data
    import numpy as np
    eigval = np.array([55.46, 4.794, 1.148])
    eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                       [-0.5808, -0.0045, -0.8140],
                       [-0.5836, -0.6948, 0.4203]])
    shape_ = data_shape


    import Myaugmentation

    aug_list_test = [
                    mx.image.ForceResizeAug(size=(shape_+int(0.1*shape_), shape_+int(0.2*shape_))),
                    mx.image.CenterCropAug((shape_, shape_)),
                    mx.image.CastAug(),

                    
    ]

    aug_list_train = [
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

    train_iter = mx.image.ImageIter(batch_size=batch_size,
                                    data_shape=(3, shape_, shape_),
                                    label_width=1,
                                    aug_list=aug_list_train,
                                    shuffle=True,
                                    path_root='',
                                    path_imglist=os.getcwd()+'/fake.lst',
                                    )
    val_iter = mx.image.ImageIter(batch_size=batch_size,
                                  data_shape=(3, shape_, shape_),
                                  label_width=1,
                                  shuffle=False,
                                  aug_list=aug_list_test,
                                  path_root='',
                                  path_imglist=os.getcwd()+'/val.lst',
                                 )

    return (custom_iter(train_iter), custom_iter(val_iter))


