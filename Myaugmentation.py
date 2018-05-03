import cv2
import mxnet as mx
from mxnet.image import  Augmenter
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
    rotated = cv2.warpAffine(image, M, (w, h))
    rotated = mx.nd.array(rotated,dtype=np.uint8)
    # 返回旋转后的图像
    return rotated

def SaltAndPepper(src,percent):
    '''
    function add salt noise
    :param src:
    :param percet:
    :return:

    '''
    Salted=src
    image=int(percent*src.shape[0]*src.shape[1])
    for i in range(image):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        if random.randint(0,1)==0:
            Salted[randX,randY]=255.
    return Salted




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
            angle=random.randint(-self.maxangel,self.maxangel)
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
class NormalizeAUG(Augmenter):

    def __init__(self, ):
        super(NormalizeAUG, self).__init__()

    def __call__(self, src):
        src = src/255.
        normalized = mx.image.color_normalize(src,
                                      mean=mx.nd.array([0.485, 0.456, 0.406]),
                                      std=mx.nd.array([0.229, 0.224, 0.225]))
        return normalized
