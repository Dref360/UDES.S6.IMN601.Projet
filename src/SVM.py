from sklearn.svm import SVC

from src.base_classifier import BaseClassifier,reduce


class SVM(BaseClassifier):
    def __init__(self):
        super().__init__()
        self._classifier = SVC()
        self._params = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'C': [0.1, 0.5, 0.75, 1],
        }

    def preprocess(self,datas):
        return reduce(lambda acc,x : acc + [x.reshape([x.shape[0],-1])],datas,[])
