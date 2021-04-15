import numpy as np
import os
import sys

import mnist_helper as m_h

from scipy import linalg

# Methods

"""
Input: X as n x d, Y as 0/1 n x k, lmbda > 0 as regularization factor
(Modify lmbda in mnist_helper constants)
Output: W-hat predictor
"""
def train(X, Y, lmbda=m_h.lmbda):
  print("train")

  assert (len(X) > 0), "X row count sanity check"

  # d-dimensional since we have (d x n) x (n x d) = d x d
  inverse_term = X.T.dot(X) + (m_h.lmbda * np.eye(len(X[0])))

  return linalg.solve(inverse_term, X.T.dot(Y))

"""
Input: W as d x k, X as m x d
Output: m-length vector with ith entry equal to the maximizer
according to each example from input X
"""
def predict(W, X):
  print("predict")

  return np.argmax(np.dot(X, W), axis=1)

def main(run_type):
  # labels are in list form, ohe are in ohe form
  X_train, labels_train, ohe_train, X_test, labels_test, ohe_test = m_h.load_dataset()

  W_train = train(X_train, ohe_train)
  
  train_predictions = predict(W_train, X_train)
  test_predictions = predict(W_train, X_test)

  train_error = m_h.calculate_error(train_predictions, labels_train)
  test_error = m_h.calculate_error(test_predictions, labels_test)

  print(train_error)
  print(test_error)

if __name__ == '__main__':
  main(sys.argv[1:])
