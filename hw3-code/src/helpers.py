# constants.py
# Applicable constants for HW3

import numpy as np
import matplotlib.pyplot as plt

import constants as c

def generate_data(n=30, d=30, k=100, sd=1):
  print("generating data")

  f_star = lambda X: 4 * np.sin(np.pi * X) * np.cos(6 * np.pi * X ** 2)

  np.random.seed(1234)

  X = np.random.uniform(0, 1, (n, ))
  error = np.random.normal(0, 1, (n, ))
  Y = f_star(X) + error

  print(Y)

  return X, Y

def plot_multiple(title, og_points, data_list, label_list):
  print("plotting fit")