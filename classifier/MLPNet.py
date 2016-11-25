from classifier.deeplearning_classifier import BaseDeepLearning
from keras.layers import Dense,Dropout
from keras.models import Sequential
from util.image_utils import linearize


class MLPNet(BaseDeepLearning):
    def _build_model(self,init_func):
        model = Sequential()
        model.add(Dense(30,activation="relu",input_shape=self.input_shape))
        model.add(Dense(self.output_size,activation="sigmoid"))
        return model

    def preprocess(self, datasets):
        return linearize(datasets)
