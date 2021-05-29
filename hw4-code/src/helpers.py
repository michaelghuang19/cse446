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


def plot_single(data, title):
  print("plotting objective functions")

  iterations = list(range(1, len(data) + 1))

  plt.plot(iterations, )

  plt.title("objective function over time")
  plt.xlabel("iterations")
  plt.ylabel("objective function")

  plt.savefig(title)


def plot_loss(data, title):
  print("plotting loss")

  for item in data:
    plt.plot(k_set, item)

  plt.title("loss over time")
  plt.xlabel("iterations")
  plt.ylabel("loss")
  plt.legends(["test, training"])

  plt.savefig(title)
