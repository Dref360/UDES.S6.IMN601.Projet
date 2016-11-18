from abc import ABCMeta, abstractmethod


class BaseClassifier:
    __metaclass__ = ABCMeta

    def __init__(self):
        self._classifier = None
        self._params = {}

    @abstractmethod
    def preprocess(self, datasets):
        pass

    def get_classifier(self):
        return self._classifier

    def get_params(self):
        return self._params
