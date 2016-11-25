from keras.layers import Dense
from keras.models import Sequential

from classifier.deeplearning_classifier import BaseDeepLearning
from util.image_utils import linearize


class MLPNet(BaseDeepLearning):
    def __call__(self, optimizer="rmsprop", init="glorot_uniform"):
        optimizer = BaseDeepLearning.get_optimizer(optimizer)
        model = Sequential()
        model.add(Dense(30, input_shape=self.input_shape))
        model.add(Dense(30))
        model.add(Dense(self.output_size))
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def preprocess(self, datasets):
        return linearize(datasets)
