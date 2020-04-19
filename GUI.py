# 前端文件

import cv2
import re
import sys
import os
import numpy as np
from GrabCut import *


def do_nothing(event, x, y, flags, param):
    pass


class ImageProcessor:
    def __init__(self, image):
        self.first_point_drawn = False
        self.second_point_drawn = False
        self.first_point = (0, 0)
        self.second_point = (0, 0)
        self.mouse_point = (0, 0)
        self.image = image
        self.new_image = image.copy()
        self.mask = np.zeros(image.shape, np.uint8)

    def checker(self, event, x, y, flags, param):
        if self.second_point_drawn:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.first_point_drawn = True
            self.first_point = (x, y)
        if self.first_point_drawn and event == cv2.EVENT_LBUTTONUP:
            self.second_point_drawn = True
            self.second_point = (x, y)
        if self.first_point_drawn and event == cv2.EVENT_MOUSEMOVE:
            self.mouse_point = (x, y)
            self.new_image = self.image.copy()
            cv2.rectangle(self.new_image, self.first_point, self.mouse_point, (255, 0, 0), 2)

    def reset(self):
        self.__init__(self.image)

    def get_rect(self):
        print("drag left mouse button and draw a rectangle to cover your target object")
        cv2.setMouseCallback('input', self.checker)
        while True:
            cv2.imshow('input', np.asarray(self.new_image, dtype=np.uint8))
            cv2.waitKey(1)
            if self.second_point_drawn:
                self.new_image = self.image.copy()
                cv2.rectangle(self.new_image, self.first_point, self.second_point, (255, 0, 0), 2)
                print("start iteration? Y/N")
                while True:
                    cv2.imshow('input', np.asarray(self.new_image, dtype=np.uint8))
                    k = 0xff & cv2.waitKey(1)
                    if k == 27 or k == ord('n'):
                        self.reset()
                        break
                    elif k == ord('y'):
                        cv2.setMouseCallback('input', do_nothing)
                        fp = (min(self.first_point[0], self.second_point[0]),
                              min(self.first_point[1], self.second_point[1]))
                        sp = (max(self.first_point[0], self.second_point[0]),
                              max(self.first_point[1], self.second_point[1]))
                        return fp, sp


def main():
    cv2.namedWindow('output')
    cv2.namedWindow('input')
    argc = len(sys.argv)
    if argc < 2:
        raise ImportError('More args needed')
    image_route = sys.argv[1]
    if not os.path.isfile(image_route):
        raise ImportError("Not a valid image")
    try:
        image = cv2.imread(image_route, cv2.IMREAD_COLOR)
    except AttributeError:
        raise ImportError("Not a valid image")

    engine = GCEngine(image)

    print(np.asarray(image, dtype=np.uint8))
    ip = ImageProcessor(image)
    temp = ip.get_rect()
    output = engine.original_iterate(temp[0], temp[1])
    print("Press F to mark foreground pixel, B to mark background pixel")
    print("Press R to rerun model")
    print("Marking foreground pixel ... ")

    while True:
        pass

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
