from functools import reduce
from sklearn.svm import SVC

from classifier.base_classifier import BaseClassifier


class SVM(BaseClassifier):

    def __init__(self, params):
        super().__init__()
        self._classifier = SVC()
        self._params = params

    def preprocess(self, datasets):
        return reduce(lambda acc, x: acc + [x.reshape([x.shape[0], -1])], datasets, [])
