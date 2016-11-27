from util.image_utils import linearize
from sklearn.svm import SVC

from classifier.base_classifier import BaseClassifier


class SVM(BaseClassifier):

    def __init__(self):
        super().__init__()
        self._classifier = SVC()

    def preprocess(self, datasets):
        return linearize(datasets)
