# helpers.py
# Applicable helpers for HW3

import numpy as np
import matplotlib.pyplot as plt

import constants as c

def generate_data(n):
  print("generating data")

  f_star = lambda x: 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)

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

def plot_acc(dataset, legends, filename, set_type, epochs=12):
  
  epoch_list = list(range(1, epochs + 1))
  
  for item in dataset:
    plt.plot(epoch_list, item)

  plt.title(set_type + " accuracy over time")
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  
  plt.legend(legends)

  plt.savefig(filename)
