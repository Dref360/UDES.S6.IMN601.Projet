from keras.layers import Dense
from keras.models import Sequential

from classifier.deeplearning_classifier import BaseDeepLearning
from util.image_utils import linearize


class MLPNet(BaseDeepLearning):
    def __call__(self, optimizer="rmsprop", init="glorot_uniform", activation="relu"):
        optimizer = BaseDeepLearning.get_optimizer(optimizer)
        model = Sequential()
        model.add(Dense(512, input_shape=self.input_shape, activation=activation, init=init))
        model.add(Dense(512, activation=activation, init=init))
        model.add(Dense(256, activation=activation, init=init))
        model.add(Dense(64, activation=activation, init=init))
        model.add(Dense(self.output_size, activation="softmax"))
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def preprocess(self, datasets):
        return linearize(datasets)
