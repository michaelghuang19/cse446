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

  X_train = df_train.iloc[:, 1:].values
  Y_train = df_train.iloc[:, :1].values

  X_test = df_test.iloc[:, 1:].values
  Y_test = df_test.iloc[:, :1].values

  input_variables = ["agePct12t29", "pctWSocSec",
                     "pctUrban", "agePct65up", "householdsize"]

  input_indices = [df_train.columns.get_loc(col_name) for col_name in input_variables]

  lamb_data = []
  nonzero_data = []

  w_list = []
  b_list = []

  lamb = h.min_lamb(X_train, Y_train)

  crime_train_lasso = lasso.Lasso(X_train, Y_train)
  crime_test_lasso = lasso.Lasso(X_test, Y_test)

  w_zero = None

  while True:
    print("crime lasso lambda value: " + str(lamb))

    if w_zero is None:
      w_zero, b = crime_train_lasso.coord_desc(lamb)
      w = w_zero
    else:
      w, b = crime_train_lasso.coord_desc(lamb, w=w_zero)

    print(w[0])
    w_list.append(np.copy(w))
    b_list.append(b)

    lamb_data.append(lamb)
    nz_count = np.count_nonzero(w)
    nonzero_data.append(nz_count)

    if lamb < 0.01:
      break

    lamb = lamb / 2

  data_list = []
  for index in input_indices:
    data_list.append([w_list[i][index - 1] for i in range(len(w_list))])

  # print(data_list)

  h.plot_single("Nonzero Coefficients over Lambda", "A5c.png",
                "Lambda", "Nonzero Coefficients", lamb_data, nonzero_data, True)
  
  h.plot_multiple("Nonzero Coefficients over Lambda", "A5d.png", "Lambda", "Weights", lamb_data, data_list, input_variables, True)

  crime_train_lasso.get_sqerror
  crime_test_lasso.get_sqerror

  # h.plot_function("Squared Error over Lambda", "A5e.png", "Lambda", "Squared Error", train_data, test_data)

  lamb = 30
  if lamb == 30:
    print("crime lasso lambda value: " + str(lamb))

    w30_train, b30_train = crime_train_lasso.coord_desc(lamb)
    w30_test, b30_test = crime_train_lasso.coord_desc(lamb)

    # find the max index locations

if __name__ == "__main__":
  main()
