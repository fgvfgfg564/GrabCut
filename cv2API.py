import cv2
import numpy as np


class CV2API:
    def __init__(self, image):
        self.image = image

    def run(self, input):
        shape = self.image.shape
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (10, 0, shape[0]-1, shape[1]-1)
        mask = ((input+1)%3).astype(np.uint8)
        print(mask.dtype,shape)
        cv2.grabCut(self.image, mask, rect, bgdModel, fgdModel, 2)
        res = np.where((mask == 2) | (mask == 0), 1, 0).astype('uint8')
        return res
