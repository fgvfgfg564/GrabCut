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
        self.masked_image = image
        self.new_image = image.copy()
        self.mask = np.ones(image.shape, np.uint8)
        self.bottom_image = np.ndarray(image.shape, np.uint8)
        self.engine = GCEngine(image)

        for i in range(self.bottom_image.shape[0]):
            for j in range(self.bottom_image.shape[1]):
                col = (i // 10 + j // 10) & 1
                col *= 255
                self.bottom_image[i][j][0] = col
                self.bottom_image[i][j][1] = col
                self.bottom_image[i][j][2] = col

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

    def original_iteration(self):
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
                        self.mask = self.engine.original_iterate(fp, sp)
                        return

    def line_drawer(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON and event == cv2.EVENT_MOUSEMOVE):
            if self.mode == 0:
                cv2.circle(self.new_image, (x, y), 5, (255, 0, 0), -1)
                cv2.circle(self.f, (x, y), 5, 1, -1)
                cv2.circle(self.b, (x, y), 5, 0, -1)
            else:
                cv2.circle(self.new_image, (x, y), 5, (0, 0, 255), -1)
                cv2.circle(self.f, (x, y), 5, 0, -1)
                cv2.circle(self.b, (x, y), 5, 1, -1)

    def modify(self):
        self.masked_image = np.where(self.mask, self.image, self.image // 4 * 3 + self.bottom_image // 4)
        self.new_image = self.masked_image.copy()
        self.f = np.zeros(self.image.shape[:2], np.uint8)
        self.b = np.zeros(self.image.shape[:2], np.uint8)
        self.mode = 0

        cv2.setMouseCallback('input', self.line_drawer)
        while True:
            cv2.imshow('input', self.new_image)
            k = 0xff & cv2.waitKey(1)
            if k == ord('r'):
                self.engine.add_foreground(self.f)
                self.engine.add_background(self.b)
                self.mask = self.engine.rerun()
                self.masked_image = np.where(self.mask, self.image, self.image // 4 * 3 + self.bottom_image // 4)
                self.new_image = self.masked_image.copy()
                self.f = np.zeros(self.image.shape[:2], np.uint8)
                self.b = np.zeros(self.image.shape[:2], np.uint8)

            if k == ord('f'):
                self.mode = 0
            if k == ord('b'):
                self.mode = 1


def main():
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

    print(type(image))
    print(np.asarray(image, dtype=np.uint8))
    ip = ImageProcessor(image)
    ip.original_iteration()
    print("Press F to mark foreground pixel, B to mark background pixel")
    print("Press R to rerun model, Press X to save and quit")
    print("Marking foreground pixel ... ")
    ip.modify()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
