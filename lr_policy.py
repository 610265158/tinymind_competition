
import mxnet as mx
from mxnet.lr_scheduler import *

class fuckingScheduler(LRScheduler):
    """ Reduce the learning rate by given a list of steps.

    Calculate the new learning rate by::

       base_lr * (1-nup/max_nup)^pwr
       if nup < max_nup, 0 otherwise.

    Parameters
    ----------
       max_update: maximum number of updates before the decay reaches 0.
       base_lr:    base learning rate
       pwr:   power of the decay term as a funtion of the current number of updates.

    """

    def __init__(self, max_update, base_lr=0.01, pwr=2):
        super(fuckingScheduler, self).__init__(base_lr)
        assert isinstance(max_update, int)
        if max_update < 1:
            raise ValueError("maximum number of updates must be strictly positive")
        self.base_lr_orig = self.base_lr
        self.max_update = max_update
        self.power = pwr
        self.base_lr = self.base_lr_orig

    def __call__(self, num_update):
        if num_update <= self.max_update:
            self.base_lr = self.base_lr_orig * pow(1.0 - float(num_update) / float(self.max_update),
                                                   self.power)

        return self.base_lr