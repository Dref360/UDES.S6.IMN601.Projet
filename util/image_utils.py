import os

import cv2
import numpy as np
from functools import reduce
from tqdm import tqdm

from util.pickle_utils import save_obj, load_obj


def create_bow(datasets, vocab_size=100):
    """
    Get the BOW vector for datasets
    :param datasets: List of dataset, [data]
    :param vocab_size: Number of features to extract
    :return:BOW vector for each dataset [[nbImage,nbVocab]]
    """
    name = "vocab{}-{}-{}".format(vocab_size, datasets[0].shape[0], datasets[1].shape[0])
    sift = cv2.xfeatures2d.SIFT_create()
    if os.path.exists("obj/{}.pkl".format(name)):
        vocab = load_obj(name)
    else:
        bow_km = cv2.BOWKMeansTrainer(vocab_size)
        for img in tqdm(np.concatenate(datasets)):
            h = sift.detectAndCompute(img, None)
            if h[1] is not None:
                bow_km.add(np.array(h[1]))
        vocab = bow_km.cluster()
        save_obj(vocab, name)

    bow = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
    bow.setVocabulary(vocab)
    bows = []
    for data in datasets:
        acc = []
        for img in tqdm(data):
            key, d = sift.detectAndCompute(img, None)
            desc = bow.compute(img, key)
            if desc is None:
                acc.append(np.zeros([vocab_size]))
            else:
                acc.append(desc[0])
        bows.append(np.asarray(acc))
    return bows


def rgb_2_gray(datasets):
    return [np.asarray([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in data]).transpose([0, 1, 2]) for data in datasets]


def linearize(datasets):
    return reduce(lambda acc, x: acc + [x.reshape([x.shape[0], -1])], datasets, [])