# pca.py
# Applicable helpers for HW4

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import constants as c
import helpers as h

from mnist import MNIST
from scipy import linalg

def part_a(X_train, X_test):
  x = x.view(-1, 28 * 28)

  mu = np.mean(X_train, axis=0)

# def part_b():
  

def main():
  X_train, X_test = h.load_mnist()



if __name__ == "__main__":
  main()
