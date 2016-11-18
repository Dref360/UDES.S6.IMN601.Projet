import argparse
import importlib
import json
import logging

import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import *

from classifier.SVM import *
from util.image_utils import rgb_2_gray, create_bow

# Note: keras library is imported dynamically


# Parse args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--db', default='mnist', type=str, help='Keras dataset to use [mnist, cifar10]')
parser.add_argument("--method", default="SVM", type=str, help="[SVM,KNN,CNN,MLP]")
parser.add_argument("--train_set_prop", default=1, type=float, help="proportion of training samples to keep")
parser.add_argument("--test_set_prop", default=1, type=float, help="proportion of test samples to keep")
parser.add_argument("--features", default="original", type=str, help="[original,BOW]")
parser.add_argument("--vocab_length", default=100, type=int, help="length of vocabulary for BOW")
parser.add_argument("--n_jobs", default=2, type=int, help="number of threads executing grid search")
options = parser.parse_args()

# Configure logger to write to console and to a file
logger = logging.getLogger('IMN601 - Project')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
for h in [logging.FileHandler('logging.log'), logging.StreamHandler()]:
    h.setFormatter(formatter)
    logger.addHandler(h)
logger.info("======= Session start =======")
logger.info('Args: %s', vars(options))

# Create classifier
config = json.load(open("config.json"))
assert options.method in ["SVM"], "Unavailable model"
if options.method == "SVM":
    classifier = SVM(config["SVM"])
else:
    classifier = None

# Load dataset based on database name
logger.info("Downloading {} Keras dataset".format(options.db))
module = importlib.import_module('keras.datasets.' + options.db)
(X_train, y_train), (X_test, y_test) = module.load_data()
# TODO: understand this
if options.db == "cifar10":
    X_train = np.transpose(X_train, [0, 2, 3, 1])
    X_test = np.transpose(X_test, [0, 2, 3, 1])

# Shrink training set size if needed
X_train = X_train[0:round(X_train.shape[0] * options.train_set_prop)]
y_train = y_train[0:round(y_train.shape[0] * options.train_set_prop)]
X_test = X_test[0:round(X_test.shape[0] * options.test_set_prop)]
y_test = y_test[0:round(y_test.shape[0] * options.test_set_prop)]

# Extract features
assert options.features in ["original", "BOW"], "Unavailable features selections"
if options.features == "BOW":
    if X_train.shape[-1] != 1 and len(X_train.shape) > 3:
        logger.info("Converting rgb images to grayscale")
        X_train, X_test = rgb_2_gray([X_train, X_test])
    logger.info("Creating bag of words")
    X_train, X_test = create_bow([X_train, X_test], options.vocab_length)
else:
    X_train, X_test = X_train / 255., X_test / 255.

# Preprocess samples
logger.info("Preprocessing samples")
X_train, X_test = classifier.preprocess([X_train, X_test])
y_train = y_train.reshape([-1])
y_test = y_test.reshape([-1])
logger.info("X_train shape : {}".format(X_train.shape))
logger.info("X_test shape : {}".format(X_test.shape))

# Perform grid-search on hyper-parameters
logger.info("Tuning hyper-parameters with grid search\n")
clf = GridSearchCV(classifier.get_classifier(), classifier.get_params(), n_jobs=options.n_jobs, verbose=4)
grid_result = clf.fit(X_train, y_train)

# Compute predictions
logger.info("Compute predictions\n")
y_true, y_pred = y_test, clf.predict(X_test)

# Log grid search results
logger.info("Grid search results\n\n" + "".join("%f (±%f) with %r" % (scores.mean(), scores.std(), params) + "\n"
                                                for params, mean_score, scores in grid_result.grid_scores_))
logger.info("Best hyper-parameters: {}\n".format(grid_result.best_params_))

# Analyse results
logger.info("Classification report\n\n{}".format(classification_report(y_true, y_pred)))
logger.info("Confusion matrix\n\n{}\n".format(confusion_matrix(y_true, y_pred)))
logger.info("Overall accuracy : ==> {} <==".format(accuracy_score(y_true, y_pred)))
logger.info("With hyper-parameters: {}".format(grid_result.best_params_))
