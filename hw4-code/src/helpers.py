# helpers.py
# Applicable helpers for HW3

import pandas as pd
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

  return X_train, X_test

"""
Helper function for plotting multiple functions
"""
def plot_multiple(plt_title, img_title, x_label, y_label, data_list, legend_list):
  
  assert (len(data_list) > 0)

  iterations = list(range(1, len(data_list[0]) + 1))

  for data in data_list:
    plt.plot(iterations, data)

  plt.title(plt_title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(legend_list)

  plt.savefig(c.results_path + img_title + c.png_exten)
  plt.close()
