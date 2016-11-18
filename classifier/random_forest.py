from util.image_utils import linearize
from sklearn.ensemble import RandomForestClassifier

from classifier.base_classifier import BaseClassifier


class RandomForest(BaseClassifier):

    def __init__(self):
        super().__init__()
        self._classifier = RandomForestClassifier()

    def preprocess(self, datasets):
        return linearize(datasets)
