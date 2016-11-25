from keras.layers import Activation, Convolution2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential

from classifier.deeplearning_classifier import BaseDeepLearning
from util.image_utils import linearize


class CNN(BaseDeepLearning):
    def __call__(self, optimizer="rmsprop", init="glorot_uniform"):
        # "Does not work yet"
        raise RuntimeError

        optimizer = BaseDeepLearning.get_optimizer(optimizer)
        model = Sequential()

        model.add(Convolution2D(16, 3, 3, border_mode="same", input_shape=self.input_shape))
        model.add(Activation("relu"))
        model.add(Convolution2D(16, 3, 3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(32, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Convolution2D(32, 3, 3))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dense(10))
        model.add(Activation("softmax"))

        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def preprocess(self, datasets):
        return linearize(datasets)
