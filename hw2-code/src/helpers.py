# Applicable helpers for HW2

import matplotlib.pyplot as plt
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

"""
Helper function for plotting function with training/testing distinction
"""
def plot_function(plt_title, img_title, x_label, y_label, train_data, test_data):
  print("plot " + plt_title)

  plt.plot(train_data)
  plt.plot(test_data)

  plt.title(plt_title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(["Training", "Testing"])

  plt.savefig(c.results_path + img_title + c.png_exten)
  plt.close()

"""
Helper function for plotting single function
"""
def plot_single(plt_title, img_title, x_label, y_label, x_data, y_data, log_scale=False):
  print("plot " + plt_title)

  plt.plot(x_data, y_data)

  plt.title(plt_title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)

  if log_scale:
    plt.xscale('log')

  plt.savefig(c.results_path + img_title + c.png_exten)
  plt.close()

"""
Helper function for getting min lasso lambda where w is all zero
"""
def min_lamb(X, Y):
  y_diff = Y.T - np.mean(Y)
  return np.max(2 * np.abs(y_diff.dot(X)))

