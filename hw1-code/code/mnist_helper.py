import numpy as np

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
  mndata = MNIST('../data/')
  X_train, labels_train = map(np.array, mndata.load_training())
  X_test, labels_test = map(np.array, mndata.load_testing())
  X_train = X_train/255.0
  X_test = X_test/255.0

  # convert labels to one-hot encoding
  ohe_train = np.zeros((len(labels_train), 10))
  ohe_test = np.zeros((len(labels_test), 10))

  for train_val in labels_train:
    ohe_train[train_val] = 1

  for test_val in labels_test:
    ohe_test[test_val] = 1

  return X_train, labels_train, ohe_train, X_test, labels_test, ohe_test

"""
Helper function for calculating the error
"""
def calculate_error(calculated_values, actual_values):
    assert (len(calculated_values) == len(actual_values)
            ), "Equal prediction list length sanity check"
    
    # convert to match boolean list
    match_summary = [calc_val == actual_values[calc_idx]
                     for calc_val, calc_idx in enumerate(calculated_values)]

    return 1 - (sum(match_summary) / len(match_summary))

