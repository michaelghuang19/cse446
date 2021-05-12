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
  
  # kinda messy in terms of individual elements vs matrices
  kf_poly = lambda x, z, d: (1 + x * z) ** d
  kf_rbf = lambda x, z, gamma: np.exp(-gamma * (x - z ** 2))
  # kf_poly = lambda x, z, d: (1 + (x.T).dot(z)) ** d
  # kf_rbf = lambda x, z, gamma: np.exp(-gamma * (np.linalg.norm(x - z, 2) ** 2))

  k_poly = Kernel(X, Y, kf_poly)
  k_rbf = Kernel(X, Y, kf_rbf)

class Kernel:
  def __init__(self, X, Y, kernel_func=None, hyperparameter=None, lambda_reg=None):
    self.X = X
    self.Y = Y
    self.kf = kernel_func
    self.hp = hyperparameter
    self.lamb = lambda_reg
    self.kmat = None

    if self.hp is None and self.lamb is None:
      print("setting best hyperparameter, lambda")
      self.loo_cv()

      print(self.hp)
      print(self.lamb)

  def loo_cv(self):
    n = len(self.X)

    error_matrix = np.zeros((len(c.hp_list), len(c.lamb_list)))

    # literally try every combo and keep track
    for i, hp in enumerate(c.hp_list):
      for j, lamb in enumerate(c.lamb_list):
        for k in range(n):
          X_val = self.X[k]
          Y_val = self.Y[k]
          
          # remove the loo val
          X_train = np.concatenate((self.X[:k], self.X[k + 1:]))
          Y_train = np.concatenate((self.Y[:k], self.Y[k + 1:]))

          f_opt = self.kernel_rr(X_train, Y_train, hp, lamb)

          error_matrix[i][j] += ((Y_val - f_opt(X_val)) ** 2)

    # find minimum index loc
    min_idx = np.argwhere(error_matrix == np.amin(error_matrix))
    min_i = min_idx[0][0]
    min_j = min_idx[0][1]

    print(error_matrix)
    print(error_matrix[min_i][min_j])
    #  == np.amin(error_matrix)

    self.hp = c.hp_list[min_i]
    self.lamb = c.lamb_list[min_j]
  
  def kernel_rr(self, X_train, Y_train, hp, lamb):
    n = len(self.X)
    # since we leave one out
    n = n - 1

    # kmat = condensed_kf(X_train, Y_train)
    kmat = np.fromfunction(lambda i, j: self.kf(X_train[i], Y_train[j], hp), shape=(n, n), dtype=int)

    # use optimizer, train
    alpha_opt = np.linalg.solve(kmat + lamb * np.eye(n), Y_train)

    f_opt = lambda x: np.sum(alpha_opt.dot(self.kf(x, X_train, hp)))

    return f_opt

if __name__ == "__main__":
  main()
