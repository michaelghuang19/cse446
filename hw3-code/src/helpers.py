# constants.py
# Applicable constants for HW3

import numpy as np

import constants as c

def generate_data(n=30, d=1000, k=100, sd=1):
  print("generating data")

  w = list(range(1, k + 1))
  w.extend(np.zeros(d - len(w)))
  w = np.expand_dims(w, axis=1)
  w = w / k

  X = np.random.normal(size=(n, d))
  offset = np.random.normal(scale=sd, size=(n, ))
  Y = (w.T).dot(X.T) + offset

  return X, np.squeeze(Y)

