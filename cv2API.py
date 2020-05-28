import cv2
import numpy as np


class CV2API:
    def __init__(self, image):
        self.image = image

    def run(self, input):
        shape = self.image.shape
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (0, 0, shape[0] - 1, shape[1] - 1)
        mask = ((input + 1) % 3).astype(np.uint8)
        print(mask.dtype, shape)
        cv2.grabCut(self.image, mask, rect, bgdModel, fgdModel, 2)
        res = np.where((mask == 2) | (mask == 0), 1, 0).astype('uint8')
        return res

    def OriginalIterate(self, p1, p2):
        shape = self.image.shape
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (p1[0], p1[1], p2[0], p2[1])
        print(rect)
        mask = np.zeros(shape[:2], np.uint8)
        print(mask.dtype, shape)
        cv2.grabCut(self.image, mask, rect, bgdModel, fgdModel, 2, mode=cv2.GC_INIT_WITH_RECT)
        res = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return res
