# A5
# Application of Lasso to crime data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import constants as c
import helpers as h
import lasso
 
# from scipy import


def main():
  print("binary logistic regression")

  df_train, df_test = h.load_crime()
  
  print(df_train.head())
  print(df_train.iloc[:,1:])

  # response y is the rate of violent crimes reported per capita in a community.

  # 95 features
  # 1595 training samples
  # 399 test samples

  X_train = df_train.iloc[:, 1:].values
  Y_train = df_train.iloc[:, :1].values
  
  X_test = df_test.iloc[:, 1:].values
  Y_test = df_test.iloc[:, :1].values

  input_variables = ["agePct12t29", "pctWSocSec",
                     "pctUrban", "agePct65up", "householdsize"]

  crime_lasso = lasso.Lasso()

if __name__ == "__main__":
  main()
