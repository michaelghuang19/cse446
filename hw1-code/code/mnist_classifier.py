import numpy as np
import os
import sys

from mnist import MNIST
# from scipy import 

"""
Constants
"""

lmbda = 0.0001

"""
Methods
"""

def load_dataset():
  mndata = MNIST('../data/')
  X_train, labels_train = map(np.array, mndata.load_training())
  X_test, labels_test = map(np.array, mndata.load_testing())
  X_train = X_train/255.0
  X_test = X_test/255.0

def train():
  print("train")

  return lmbda

def predict():
  print("predict")

  return 0

def main(run_type):
  load_dataset()

  print(train())

  print(predict())

if __name__ == '__main__':
  main(sys.argv[1:])
