from sklearn.svm import SVC

from src.base_classifier import BaseClassifier, reduce


class SVM(BaseClassifier):
    def __init__(self, config):
        super().__init__()
        self._classifier = SVC()
        self._params = config.get_config("SVM")

    def preprocess(self, datas):
        return reduce(lambda acc, x: acc + [x.reshape([x.shape[0], -1])], datas, [])
