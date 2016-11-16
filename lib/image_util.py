import cv2
import numpy as np


def get_SIFT(datas):
    acc = []
    sift = cv2.xfeatures2d.SIFT_create()
    for d in datas:
        kp, desc = sift.detectAndCompute(d, None)
        acc.append(desc)
    return np.asarray(acc)
