# A4
# Lasso implementation

import matplotlib.pyplot as plt
import numpy as np

import constants as c
import helpers as h

# from scipy import

def main():
  print("binary logistic regression")

  lasso = Lasso(c.reg_lambda)

  lasso.generate_synthetic_data()

  lamb = h.min_lamb(lasso.X, lasso.Y)



class Lasso:
  def __init__(self, reg_lambda):
    self.lamb = reg_lambda


  def lasso(self, cutoff):
    print("lasso")

    # while max(np.max(np.abs(grad_w)), grad_b) < cutoff:

  def generate_synthetic_data(self, n=500, d=1000, k=100, sd=1):
    print("generating synthetic data")

    w = list(range(1, k + 1))
    w.extend(np.zeros(d - len(w)))
    w = np.expand_dims(w, axis=1)
    w = w / k
    
    X = np.random.normal(size=(n, d))

    offset = np.random.normal(scale=sd, size=n)
    Y = (w.T).dot(X.T) + offset

    self.X = X
    self.Y = Y.T

if __name__ == "__main__":
  main()
