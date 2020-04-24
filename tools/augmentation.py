import numpy as np
from PIL import Image

class RandomRemoval(object):
    def __init__(self, prob=0.5, intensity=0.2):
        self.prob = prob # 이미지 탈락이 발생하는 확률
        self.intensity = intensity # 이미지에서 탈락될 점군의 비율

    def __call__(self, img):
        # TODO: Image Level Augmentation, Point Level Augmentation하고 비교 해야 함
        #cimg = np.copy(img)
        img = np.array(img)
        foreground = (img[:,:,0] > 0) | (img[:,:,1] > 0) | (img[:,:,2] > 2)
        numForeground = np.sum(foreground, axis=None) # Sum all
        removalIndices = np.random.permutation(numForeground)
        removalCount = int(numForeground*self.intensity)
        orishape = img.shape
        fx,fy = np.where(foreground)
        removalX = fx[removalIndices[:removalCount]]
        removalY = fy[removalIndices[:removalCount]]
        img[removalX,removalY,:] = 0

        return Image.fromarray(img)
