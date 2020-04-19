# 后端文件

import cv2
import numpy as np


class GCEngine:
    def __init__(self, img):
        # to be implemented
        self.img = img
        self.img_size = img.shape

    def original_iterate(self, p1, p2):
        # to be implemented
        return np.zeros(self.img_size, np.uint8)

    def add_foreground(self, pixels):
        # to be implemented
        return np.zeros(self.img_size, np.uint8)

    def add_background(self, pixels):
        # to be implemented
        return np.zeros(self.img_size, np.uint8)
