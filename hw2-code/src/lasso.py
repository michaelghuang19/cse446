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

class Lasso:
  def __init__(self, reg_lambda):
    self.lamb = reg_lambda

if __name__ == "__main__":
  main()
