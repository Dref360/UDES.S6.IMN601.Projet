from keras.wrappers.scikit_learn import KerasClassifier

from classifier.base_classifier import BaseClassifier
from keras.optimizers import RMSprop,SGD


class BaseDeepLearning(BaseClassifier):
    def __init__(self,input_shape,output_size):
        super().__init__()
        self.input_shape = input_shape
        self.output_size = output_size
        self._classifier = KerasClassifier(build_fn=self._build_func)

    def _build_func(self, optimizer="rmsprop", init="glorot_uniform"):
        if isinstance(optimizer,tuple):
            """We expect ("name", param)"""
            name,param = optimizer
            assert name in ["rmsprop","sgd"],"Not supported optimizer"
            if name == "rmsprop":
                optimizer = RMSprop(*param)
            elif name == "sgd":
                optimizer = SGD(*param)

        model = self._build_model(init)
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        return model

    def _build_model(self,init_func):
        raise NotImplementedError

    def preprocess(self, datasets):
        return datasets
