# constants.py
# Applicable constants for HW3

import numpy as np
import matplotlib.pyplot as plt

import constants as c

def generate_data(n=30, d=30, k=100, sd=1):
  print("generating data")

  # capital X?
  f_star = lambda x: 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)

  X = np.random.uniform(0, 1, (n, ))
  error = np.random.normal(0, 1, (n, ))
  Y = f_star(X) + error

  # np.linspace(0, 1, n)

  return X, Y

def plot_multiple(title, og_points, data_list, label_list):
  print("plotting fit")
