# A4
# Lasso implementation

import matplotlib.pyplot as plt
import numpy as np

import constants as c
import helpers as h

# from scipy import

def main():
  print("binary logistic regression")

  lamb_data = []
  nonzero_data = []

  fdr_data = []
  tpr_data = []

  lasso = Lasso()

  lasso.generate_synthetic_data()
  lamb = h.min_lamb(lasso.X, lasso.Y)

  n, d = lasso.X.shape
  k = 100

  w = np.zeros((d, 1))

  # abritrarily stop when we hit 995/1000
  while np.count_nonzero(w) < d - 5:
    print("lasso lambda value: " + str(lamb))

    # perform coordinate descent 
    w, _ = lasso.coord_desc(lamb)

    # append lambda/nonzero data
    lamb_data.append(lamb)
    nz_count = np.count_nonzero(w)
    nonzero_data.append(nz_count)

    # update lambda
    lamb = lamb / 2

    # append fdr/tpr data
    if k == 0 or nz_count == 0:
      continue

    fdr_data.append(np.count_nonzero(w[k:]) / nz_count)
    tpr_data.append(np.count_nonzero(w[:k]) / k)

  h.plot_single("Nonzero Coefficients over Lambda", "A4a.png",
                "Lambda", "Nonzero Coefficients", lamb_data, nonzero_data, True)

  h.plot_single("TPR over FDR", "A4b.png",
                "FDR", "TPR", fdr_data, tpr_data)

class Lasso:
  def __init__(self, X=None, Y=None):
    self.X = X
    self.Y = Y

  def coord_desc(self, lamb, cutoff=c.cutoff, w=None):
    n, d = self.X.shape

    b = 0
    a = 2 * np.sum(np.square(self.X), axis=0)
    c = np.zeros((d, 1))

    if w == None:
      w = np.zeros((d, 1))

    while True:
      w_diff = 0

      w = np.squeeze(w)
      wx_sum = (w.T).dot(self.X.T)
      b = (1 / n) * np.sum(self.Y - wx_sum)
      
      for k in range(d):
        old_w = w[k]

        # only include where j =/= k, so subtract that
        wx_neqk_sum = (w.T).dot(self.X.T) - np.multiply(w[k], self.X[:, k])
        
        c[k] = 2 * self.X[:, k].dot(self.Y - (b + wx_neqk_sum))

        if c[k] < -lamb:
          w[k] = (c[k] + lamb) / a[k]
        elif c[k] > lamb:
          w[k] = (c[k] - lamb) / a[k]
        else: 
          w[k] = 0

        if np.abs(w[k] - old_w) > w_diff:
          w_diff = np.abs(w[k] - old_w)

      if w_diff < cutoff:
        break
    
    return w, b

  def generate_synthetic_data(self, n=500, d=1000, k=100, sd=1):
    print("generating synthetic data")

    w = list(range(1, k + 1))
    w.extend(np.zeros(d - len(w)))
    w = np.expand_dims(w, axis=1)
    w = w / k
    
    X = np.random.normal(size=(n, d))
    offset = np.random.normal(scale=sd, size=(n, ))
    Y = (w.T).dot(X.T) + offset

    self.X = X
    self.Y = np.squeeze(Y)
    

if __name__ == "__main__":
  main()
