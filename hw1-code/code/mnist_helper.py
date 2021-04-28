# mnist_helper.py

import numpy as np
import sys
import os

from mnist import MNIST

# Global Variables

lmbda = 10E-4

n = 0
m = 0

# Helper Functions

"""
Helper function for loading in MNIST data set
"""
def load_dataset():
  cur_file_dir = os.path.dirname(os.path.abspath(__file__))

  mndata = MNIST(os.path.dirname(cur_file_dir) + '/data/')
  X_train, labels_train = map(np.array, mndata.load_training())
  X_test, labels_test = map(np.array, mndata.load_testing())
  X_train = X_train/255.0
  X_test = X_test/255.0

  # convert labels to one-hot encoding
  ohe_train = np.zeros((len(labels_train), 10))
  ohe_test = np.zeros((len(labels_test), 10))

  for idx, train_val in enumerate(labels_train):
    ohe_train[idx][train_val] = 1

  for idx, test_val in enumerate(labels_test):
    ohe_test[idx][test_val] = 1

  return X_train, labels_train, ohe_train, X_test, labels_test, ohe_test

"""
Helper function for calculating the error
"""
def calculate_error(calculated_values, actual_values):
    assert (len(calculated_values) == len(actual_values)
            ), "Equal prediction list length sanity check"
    
    # convert to match boolean list
    match_summary = [calc_val == actual_values[calc_idx]
                     for calc_idx, calc_val in enumerate(calculated_values)]

    return 1 - (sum(match_summary) / len(match_summary))

