# constants.py
# Applicable constants for HW3

import numpy as np
import matplotlib.pyplot as plt

import constants as c

def generate_data(n=30, d=30, k=100, sd=1):
  print("generating data")

  f_star = lambda x: 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)

  np.random.seed(1234)

  X = np.random.uniform(0, 1, (n, ))
  error = np.random.normal(0, 1, (n, ))
  Y = f_star(X) + error

  return X, Y, f_star

def ylimit_plot(bot, top):
  plt.ylim(bot, top)

def plot_multiple(title, file_name, x, y, data_list, label_list, ylimits=None):
  print("plotting og, true, and fit")

  plt.plot(x, y, "o")
  
  for data_set in data_list:
    plt.plot(c.x_list, data_set)

  if ylimits is not None:
    ylimit_plot(ylimits[0], ylimits[1])

  plt.title(title)
  plt.xlabel("x values")
  plt.ylabel("y values")
  plt.legend(label_list)

  plt.savefig(c.results_path + file_name + c.png_exten)
  plt.close()
