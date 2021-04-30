# Applicable helpers for HW2

import numpy as np

import constants as c

from mnist import MNIST
from scipy import linalg

"""
Helper function for loading in MNIST data set
"""
def load_mnist():
  # load data
  mndata = MNIST(c.data_path)

  X_train, labels_train = map(np.array, mndata.load_training())
  X_test, labels_test = map(np.array, mndata.load_testing())
  X_train = X_train/255.0
  X_test = X_test/255.0

  # count number of 2s and 7s
  train_mask = np.logical_or(labels_train == 2, labels_train == 7)
  test_mask = np.logical_or(labels_test == 2, labels_test == 7)

  X_train = X_train[train_mask]
  X_test = X_test[test_mask]

  labels_train = labels_train[train_mask]
  labels_test = labels_test[test_mask]

  train_count = sum(labels_train == 2) + sum(labels_train == 7)
  test_count = sum(labels_test == 2) + sum(labels_test == 7)

  # convert labels to +/- 1
  one_train = np.zeros(train_count)
  one_test = np.zeros(test_count)

  for idx, train_val in enumerate(labels_train):
    one_train[idx] = c.digit_to_one[train_val]

  for idx, test_val in enumerate(labels_test):
    one_test[idx] = c.digit_to_one[test_val]

  return X_train, labels_train, one_train, X_test, labels_test, one_test




