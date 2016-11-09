import cv2
import numpy as np


def get_BOW(datas, nbVocab=100):
    """
    Get the BOW vector for datas
    :param datas: List of dataset, [data]
    :param nbVocab: Number of feature to extract
    :return:BOW vector for each dataset [[nbImage,nbVocab]]
    """
    sift = cv2.xfeatures2d.SIFT_create()
    bowKM = cv2.BOWKMeansTrainer(nbVocab)
    for img in np.concatenate(datas):
        h = sift.detectAndCompute(img, None)
        bowKM.add(np.array(h[1]))
    vocab = bowKM.cluster()
    bow = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
    bow.setVocabulary(vocab)
    bows = []
    for data in datas:
        acc = []
        for img in data:
            key, d = sift.detectAndCompute(img, None)
            desc = bow.compute(img, key)
            acc.append(desc)
        bows.append(np.asarray(acc).reshape([-1, nbVocab]))
    return bows
