from abc import ABCMeta, abstractmethod


class BaseClassifier:
    __metaclass__ = ABCMeta

    def __init__(self):
        self._classifier = None

    @abstractmethod
    def preprocess(self, datasets):
        pass

    def get_classifier(self):
        return self._classifier
