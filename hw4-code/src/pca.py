# pca.py
# Applicable helpers for HW4 A3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import constants as c
import helpers as h

from mnist import MNIST
from scipy import linalg

def part_a(X, mu=None):
  n, d = X.shape

  if mu is None:
    mu = np.mean(X, axis=0)

  diff = X - mu
  sigma = np.matmul(diff.T, diff) / n

  lamb, v = np.linalg.eig(sigma)
  lamb = lamb.real
  v = v.real

  return lamb, v

def part_c(X_train, X_test):
  train_mse_data = []
  test_mse_data = []
  frac_data = []
  
  e_train, v_train = part_a(X_train)

  for i in range(c.k):
    recon_train = (v_train[:i, ].T).dot(v_train[:i, ])
    recon_test = recon_train.dot(X_test.T).T
    recon_train = recon_train.dot(X_train.T).T

    mse_train = np.sum(np.square(recon_train - X_train)) / X_train.shape[0]
    mse_test = np.sum(np.square(recon_test - X_test)) / X_test.shape[0]
    
    train_mse_data.append(mse_train)
    test_mse_data.append(mse_test)

    frac = 1 - (np.sum(e_train[:i+1]) - np.sum(e_train))

    frac_data.append(frac)

  return train_mse_data, test_mse_data, frac_data

def part_d(v_list):
  fig, axes = plt.subplots(2, 5)

  axes_list = []
  for i, item in enumerate(axes):
    axes_list += list(item)

  for i, ax in enumerate(axes_list):
    ax.imshow(v_list[i], cmap='gray')

  plt.savefig(c.results_path + "a6_c")

def main():
  output = open(c.results_path + "a3.txt", "w")

  X_train, X_test = h.load_mnist()

  e_list, v_list = part_a(X_train)
  output.write(str([e_list[0], e_list[1], e_list[9],
                   e_list[29], e_list[49]]) + "\n")
  output.write(str(sum(e_list)) + "\n")

  train_mse_data, test_mse_data, frac_data = part_c(
      X_train, X_test)

  h.plot_multiple("error over k", "a3_cerr", "k", "error", 
                  [train_mse_data, test_mse_data], c.tt_list)
  h.plot_multiple("obj over k", "a3_cobj", "k", "obj", 
                  [frac_data], ["frac"])

  part_d(v_list)

  output.close()


if __name__ == "__main__":
  main()
