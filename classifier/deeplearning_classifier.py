from keras.optimizers import RMSprop, SGD
from keras.wrappers.scikit_learn import KerasClassifier

from classifier.base_classifier import BaseClassifier


class BaseDeepLearning(BaseClassifier):
    def __init__(self, input_shape, output_size, n_epoch):
        super().__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self._classifier = KerasClassifier(build_fn=self, nb_epoch=n_epoch)

    def __call__(self, optimizer="rmsprop", init="glorot_uniform"):
        raise NotImplementedError

    @staticmethod
    def get_optimizer(optimizer):
        if isinstance(optimizer, tuple):
            """We expect ("name", param)"""
            name, param = optimizer
            assert name in ["rmsprop", "sgd"], "Not supported optimizer"
            if name == "rmsprop":
                optimizer = RMSprop(*param)
            elif name == "sgd":
                optimizer = SGD(*param)
        return optimizer
