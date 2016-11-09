import os

import cv2
import numpy as np
from tqdm import tqdm

from lib.pickle_saving import save_obj, load_obj


def get_BOW(datas, nbVocab=100):
    """
    Get the BOW vector for datas
    :param datas: List of dataset, [data]
    :param nbVocab: Number of feature to extract
    :return:BOW vector for each dataset [[nbImage,nbVocab]]
    """
    name = "vocab{}".format(nbVocab)
    sift = cv2.xfeatures2d.SIFT_create()
    if os.path.exists("obj/{}.pkl".format(name)):
        vocab = load_obj(name)
    else:
        bowKM = cv2.BOWKMeansTrainer(nbVocab)
        for img in np.concatenate(datas):
            h = sift.detectAndCompute(img, None)
            if h[1] is not None:
                bowKM.add(np.array(h[1]))
        vocab = bowKM.cluster()
        save_obj(vocab, name)

    bow = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
    bow.setVocabulary(vocab)
    bows = []
    for data in tqdm(datas):
        acc = []
        for img in data:
            key, d = sift.detectAndCompute(img, None)
            desc = bow.compute(img, key)
            if desc is None:
                acc.append(np.zeros([nbVocab]))
            else:
                acc.append(desc[0])
        bows.append(np.asarray(acc))
    return bows


def RGB_2_gray(datas):
    return [np.asarray([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in data]).transpose([0, 1, 2]) for data in datas]
