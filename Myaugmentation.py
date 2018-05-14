import cv2
import mxnet as mx
from mxnet.image import Augmenter
import random
import numpy as np
def rotate(src, angle, center=None, scale=1.0):

    '''
    function to rotate a img based on center point
    :param src: source image
    :param angle: the angle rotated
    :param center:
    :param scale:
    :return:img
    '''
    image = src.asnumpy()
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h),flags=3,borderMode=cv2.BORDER_REPLICATE)
    rotated = mx.nd.array(rotated)

    # 返回旋转后的图像
    return rotated

def SaltAndPepper(src,percent):
    '''

    '''
    pass
    #TODO

def blur(src,kernel):
    '''
    :param src: source image
    :param kernel: the kernel applied to src
    '''
    image = src.asnumpy()
    image=cv2.GaussianBlur(image, kernel,0),
    blured = mx.nd.array(image[0])
    
    return blured


#####################
class RandomRotateAug(Augmenter):
    """Make randomrotate.

    Parameters
    ----------
    angel : float or int the max angel to rotate

    p : the possibility the img be rotated

    """
    def __init__(self, angel,possibility):
        super(RandomRotateAug, self).__init__(angel=angel)
        self.maxangel = angel
        self.p=possibility
    def __call__(self, src):
        """Augmenter body"""
        a = random.random()
        if a > self.p:
            return src
        else:
            angle=random.uniform(-self.maxangel,self.maxangel)
            return rotate(src,angle)


class RandomNoiseAug(Augmenter):
    """Make RandomNoiseAug.

    Parameters
    ----------
    percet : float [0,1]
    p : the possibility the img be rotated
    """
    def __init__(self, percent,possibility):
        super(RandomNoiseAug, self).__init__(percent=percent)
        self.percent = percent
        self.p=possibility
    def __call__(self, src):
        """Augmenter body"""
        a = random.random()
        if a > self.p:
            return src
        else:
            return SaltAndPepper(src,self.percent)

####Imagenet data Normlization
class NormalizeAug(Augmenter):

    def __init__(self, ):
        super(NormalizeAug, self).__init__()

    def __call__(self, src):
        src = src/255.
        normalized = mx.image.color_normalize(src,
                                      mean=mx.nd.array([0.485, 0.456, 0.406]),
                                      std=mx.nd.array([0.229, 0.224, 0.225]))
        return normalized



###blue aug
class BlurAug(Augmenter):
    def __init__(self,  possibility,kernel):
        super(BlurAug, self).__init__(possibility=possibility,kernel=kernel)

        self.p = possibility
        self.kernel=kernel
    def __call__(self, src):
        """Augmenter body"""
        a = random.random()
        if a > self.p:
            return src
        else:
            return blur(src,self.kernel)


###cast to int8,  similar to noise aug
class Castint8Aug(Augmenter):
    def __init__(self,  possibility):
        super(Castint8Aug, self).__init__(possibility=possibility)

        self.p = possibility

    def __call__(self, src):
        """Augmenter body"""
        a = random.random()
        if a > self.p:
            return src
        else:
            return mx.nd.array(src,dtype=np.int8)


