# kernel.py
# Applicable code for HW3 A3

import numpy as np

import constants as c
import helpers as h

to_run = {
  "abc" : False,
  "de" : True,
}

def main():

  X, Y, true_f = h.generate_data(30)

  # individual elements over entire matrix
  kf_poly = lambda x, z, d: (1 + x * z) ** d
  kf_rbf = lambda x, z, gamma: np.exp(-gamma * ((x - z) ** 2))

  if to_run["abc"]:
    """
    part (a)
    """

    k_poly = Kernel(X, Y, kf_poly)
    k_rbf = Kernel(X, Y, kf_rbf)

    """
    part (b)
    """

    true_data = [true_f(x_val) for x_val in c.x_list]
    poly_pred_data = k_poly.get_fhat_data(X, Y)
    rbf_pred_data = k_rbf.get_fhat_data(X, Y)

    poly_list = [true_data, poly_pred_data]
    rbf_list = [true_data, rbf_pred_data]

    h.plot_multiple("poly", "a3_bi", X, Y, poly_list, c.pred_labels, c.a3b_ylimits)
    h.plot_multiple("rbf", "a3_bii", X, Y, rbf_list, c.pred_labels, c.a3b_ylimits)

    """
    part (c)
    """

    poly_5, poly_95 = k_poly.bootstrap(c.B)
    rbf_5, rbf_95 = k_rbf.bootstrap(c.B)

    poly_list = [true_data, poly_pred_data, poly_5, poly_95]
    rbf_list = [true_data, rbf_pred_data, rbf_5, rbf_95]

    h.plot_multiple("poly", "a3_ci", X, Y, poly_list, c.pct_labels, c.a3b_ylimits)
    h.plot_multiple("rbf", "a3_cii", X, Y, rbf_list, c.pct_labels, c.a3b_ylimits)

  if to_run["de"]:

    """
    part (d)
    """

    # repeated a
    X, Y, true_f = h.generate_data(300)

    k_poly = Kernel(X, Y, kf_poly, kfold=True)
    k_rbf = Kernel(X, Y, kf_rbf, kfold=True)

    # repeated b
    true_data = [true_f(x_val) for x_val in c.x_list]
    poly_pred_data = k_poly.get_fhat_data(X, Y)
    rbf_pred_data = k_rbf.get_fhat_data(X, Y)

    poly_list = [true_data, poly_pred_data]
    rbf_list = [true_data, rbf_pred_data]

    h.plot_multiple("poly", "a3_d.bi", X, Y, poly_list, c.pred_labels, c.a3b_ylimits)
    h.plot_multiple("rbf", "a3_d.bii", X, Y, rbf_list, c.pred_labels, c.a3b_ylimits)

    # repeated c
    poly_5, poly_95 = k_poly.bootstrap(c.B)
    rbf_5, rbf_95 = k_rbf.bootstrap(c.B)

    poly_list = [true_data, poly_pred_data, poly_5, poly_95]
    rbf_list = [true_data, rbf_pred_data, rbf_5, rbf_95]

    h.plot_multiple("poly", "a3_d.ci", X, Y, poly_list, c.pct_labels, c.a3b_ylimits)
    h.plot_multiple("rbf", "a3_d.cii", X, Y, rbf_list, c.pct_labels, c.a3b_ylimits)

    """
    part (e)
    """

    X, Y, true_f = h.generate_data(c.m)

    poly_pred = k_poly.kernel_rr(X, Y, k_poly.hp, k_poly.lamb)
    rbf_pred = k_rbf.kernel_rr(X, Y, k_rbf.hp, k_rbf.lamb)

    bs_5, bs_95 = get_bootstrap_values(X, Y, poly_pred, rbf_pred)

    print(bs_5)
    print(bs_95)

def get_bootstrap_values(X, Y, poly_pred, rbf_pred, m=c.m, B=c.B):
  
  diff_list = np.zeros(B)

  for i in range(B):
    
    index_samples = np.random.choice(m, m)

    x_b = X[index_samples]
    y_b = Y[index_samples]

    poly_diff = (np.abs(y_b - [poly_pred(x) for x in x_b]) ** 2)
    rbf_diff = (np.abs(y_b - [rbf_pred(x) for x in x_b]) ** 2)

    diff_list[i] = (1 / m) * np.sum(poly_diff - rbf_diff)


  bs_5 = np.percentile(diff_list, 5, axis=0)
  bs_95 = np.percentile(diff_list, 95, axis=0)

  return bs_5, bs_95

class Kernel:
  def __init__(self, X, Y, kernel_func=None, hyperparameter=None, lambda_reg=None, kfold=False):
    self.X = X
    self.Y = Y
    self.kf = kernel_func
    self.hp = hyperparameter
    self.lamb = lambda_reg
    self.kmat = None

    if self.hp is None and self.lamb is None:
      print("setting best hyperparameter, lambda")

      if kfold:
        self.kfold_cv()
      else:
        self.loo_cv()
      

    print("hyperparam: " + str(self.hp))
    print("lambda: " + str(self.lamb))

  def get_fhat_data(self, x, y):
    pred_f = self.kernel_rr(x, y, self.hp, self.lamb)

    return [pred_f(x_val) for x_val in c.x_list]

  def loo_cv(self):
    n = len(self.X)

    error_matrix = np.zeros((len(c.hp_list), len(c.lamb_list)))

    # try every combo and keep track of errors
    for i, hp in enumerate(c.hp_list):
      for j, lamb in enumerate(c.lamb_list):
        for k in range(n):
          X_val = self.X[k]
          Y_val = self.Y[k]
          
          # train without the loo val
          X_train = np.concatenate((self.X[:k], self.X[k + 1:]))
          Y_train = np.concatenate((self.Y[:k], self.Y[k + 1:]))

          f_opt = self.kernel_rr(X_train, Y_train, hp, lamb)

          error_matrix[i][j] += (np.abs(f_opt(X_val) - Y_val) ** 2)
        
        error_matrix[i][j] /= n

    # find minimum index loc
    min_idx = np.argwhere(error_matrix == np.amin(error_matrix))
    min_i = min_idx[0][0]
    min_j = min_idx[0][1]

    assert (error_matrix[min_i][min_j] == np.amin(error_matrix)), "Sanity check min error is accurate"

    self.hp = c.hp_list[min_i]
    self.lamb = c.lamb_list[min_j]
  
  # TODO: refactor this into one module
  def kfold_cv(self):
    n = len(self.X)

    error_matrix = np.zeros((len(c.hp_list), len(c.lamb_list)))

    # try every combo and keep track of errors
    for i, hp in enumerate(c.hp_list):
      for j, lamb in enumerate(c.lamb_list):
        for k in range(c.num_fold):
          fold_start = k * int(n / 10)
          fold_end =  (k + 1) * int(n / 10)

          X_val = self.X[fold_start:fold_end]
          Y_val = self.Y[fold_start:fold_end]

          # train without the loo val
          X_train = np.concatenate((self.X[:fold_start], self.X[fold_end:]))
          Y_train = np.concatenate((self.Y[:fold_start], self.Y[fold_end:]))

          f_opt = self.kernel_rr(X_train, Y_train, hp, lamb)

          error_matrix[i][j] += np.sum(([f_opt(x) for x in X_val] - Y_val) ** 2)

        error_matrix[i][j] /= len(X_val)

    # find minimum index loc
    min_idx = np.argwhere(error_matrix == np.amin(error_matrix))
    min_i = min_idx[0][0]
    min_j = min_idx[0][1]

    assert (error_matrix[min_i][min_j] == np.amin(
        error_matrix)), "Sanity check min error is accurate"

    self.hp = c.hp_list[min_i]
    self.lamb = c.lamb_list[min_j]

  def kernel_rr(self, X_train, Y_train, hp, lamb):
    n = len(X_train)

    kmat = np.fromfunction(lambda i, j: self.kf(X_train[i], X_train[j], hp), shape=(n, n), dtype=int)

    # use optimizer, train
    alpha_opt = np.linalg.solve(kmat + lamb * np.eye(n), Y_train)

    pred_f = lambda x: alpha_opt.dot(self.kf(X_train, x, hp))

    return pred_f

  def bootstrap(self, iterations):
    n = len(self.X)

    B = iterations
    num_x = len(c.x_list)

    fhat_list = np.zeros((B, num_x))

    for i in range(B):
      index_samples = np.random.choice(n, n)

      x_b = self.X[index_samples]
      y_b = self.Y[index_samples]

      fhat_list[i] = np.copy(self.get_fhat_data(x_b, y_b))

    bot_5 = np.percentile(fhat_list, 5, axis=0)
    top_5 = np.percentile(fhat_list, 95, axis=0)

    return bot_5, top_5

if __name__ == "__main__":
  main()
