# kernel.py
# Applicable code for HW3 A3

import numpy as np

import constants as c
import helpers as h

def main():
  print("hello world")

  X, Y = h.generate_data()
  # shapes: (30, )

  # np.expand_dims(Y, axis=1)

  kf_poly = lambda x, z, d: (1 + (x.T).dot(z)) ** d
  kf_rbf = lambda x, z, gamma: np.exp(-gamma * (np.linalg.norm(x - z, 2) ** 2))

  k_poly = Kernel(X, Y, kf_poly)
  k_rbf = Kernel(X, Y, kf_rbf)

class Kernel:
  def __init__(self, X, Y, kernel_func=None):
    self.X = X
    self.Y = Y
    self.kf = kernel_func

    print(self.kf(1, 1, 1))

  def loo_cv(self):
    print("running leave-one-out cross validation")



if __name__ == "__main__":
  main()
