# Here, we write the code to train the model
import argparse
import logging

from keras.datasets import mnist, cifar10
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix

from lib.config_reader import ConfigReader
from src.SVM import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--db", dest="db", default="mnist", type=str, help="database to use [cifar10,mnist]")
parser.add_argument("--batch_size", dest="batch_size", default=1, type=int, help="batch size")
parser.add_argument("--n_epochs", dest="n_epochs", default=10, type=int, help="nb epochs")
parser.add_argument("--method", dest="method", default="SVM", type=str, help="[SVM,KNN,CNN,MLP]")
options = parser.parse_args()

logging.basicConfig(filename='logging.log', level=logging.DEBUG,
                    format='%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s')
logging.info(vars(options))
config = ConfigReader("config.json")
assert options.method in ["SVM"], "Unavailable model"
if options.method == "SVM":
    classifier = SVM(config)
else:
    classifier = None

assert options.db in ["mnist", "cifar10"], "Unavailable db"
if options.db == "mnist":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
else:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train, X_test = classifier.preprocess([X_train, X_test])
X_train, X_test = X_train / 255., X_test / 255.
y_train = y_train.reshape([-1])
y_test = y_test.reshape([-1])

param_grid = classifier.get_params()

scores = ['precision', 'recall']

print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(classifier.get_classifier(), param_grid, n_jobs=4, verbose=4)
grid_result = clf.fit(X_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
pr, rc, f, true_sum = precision_recall_fscore_support(y_true, y_pred, beta=2)
accuracy = accuracy_score(y_true, y_pred)
print(classification_report(y_true, y_pred))
print("--SCORE--")
print("Precision : {}, Recall : {}, F-Measure : {}".format(pr, rc, f))
print("Accuracy : {}".format(accuracy))
print("--CONFUSION MATRIX--")
print(confusion_matrix(y_true, y_pred))
