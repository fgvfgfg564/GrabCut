import cv2
import numpy as np
from GrabCut import GCEngine
from cv2API import CV2API

test_num = 27


def dist(A, B):
    return np.linalg.norm(A - B)


def test(image_route, input_route, gt_route, Method):
    print(image_route)
    try:
        image = cv2.imread(image_route)
    except AttributeError:
        raise ImportError("Not a valid image")

    try:
        inp = cv2.imread(input_route)
    except AttributeError:
        raise ImportError("Not a valid input")

    try:
        gt = cv2.imread(gt_route)
    except AttributeError:
        raise ImportError("Not a valid ground truth")

    size = image.shape
    new_input = np.zeros(size[:-1], dtype=np.int)
    new_gt = np.zeros(size[:-1])
    for i in range(size[0]):
        for j in range(size[1]):
            if inp[i][j][0] == 0:
                new_input[i][j] = 0
            elif inp[i][j][0] == 128:
                new_input[i][j] = 1
            else:
                new_input[i][j] = 2
            new_gt[i][j] = gt[i][j][0] / 255

    algorithm = Method(image)
    output = algorithm.run(new_input)
    return 1 - np.sum(np.fabs(output - new_gt)) / (size[0] * size[1])


def test_method(Method):
    ans = 0
    datas = []
    for i in range(1, test_num + 1):
        cur = test("data/input_training_lowres/GT%02d.png" % i, "data/trimap_training_lowres/Trimap1/GT%02d.png" % i,
                   "data/gt_training_lowres/GT%02d.png" % i, Method)
        ans += cur
        print("test case:", i, "score=", cur)
        datas.append(cur)
    for each in datas:
        print(each, end='\t')
    print()
    return ans / test_num


def main():
    print(test_method(CV2API))


if __name__ == "__main__":
    main()
