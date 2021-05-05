# crime-lasso.py
# A5: Application of Lasso to crime data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tabulate import tabulate

import constants as c
import helpers as h
import lasso

def main():
  print("binary logistic regression")

  df_train, df_test = h.load_crime()

  X_train = df_train.iloc[:, 1:].values
  Y_train = df_train.iloc[:, :1].values

  X_test = df_test.iloc[:, 1:].values
  Y_test = df_test.iloc[:, :1].values

  input_indices = [df_train.columns.get_loc(col_name) for col_name in c.input_variables]

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

    w_list.append(np.copy(w))
    b_list.append(b)

    lamb_data.append(lamb)
    nz_count = np.count_nonzero(w)
    nonzero_data.append(nz_count)

    if lamb < c.cutoff:
      break

    lamb = lamb / 2

  data_list = []
  for index in input_indices:
    data_list.append([w_list[i][index - 1] for i in range(len(w_list))])

  h.plot_single("Nonzero Coefficients over Lambda", "a5_c",
                "Lambda", "Nonzero Coefficients", lamb_data, nonzero_data, True)
  
  h.plot_multiple("Nonzero Coefficients over Lambda", "a5_d", "Lambda", "Weights", lamb_data, data_list, c.input_variables, True)

  train_sqerror_list = []
  test_sqerror_list = []

  w_list = np.array(w_list).T
  b_list = np.array(b_list)

  train_sqerror_list = crime_train_lasso.get_sqerror(w_list, b_list)
  test_sqerror_list = crime_test_lasso.get_sqerror(w_list, b_list)

  h.plot_function("Squared Error over Lambda", "a5_e",
                  "Lambda", "Squared Error", train_sqerror_list, test_sqerror_list, x_data=lamb_data, log_scale=True)

  lamb = 30
  if lamb == 30:
    # print("crime lasso lambda value: " + str(lamb))

    w30_train, b30_train = crime_train_lasso.coord_desc(lamb)
    w30_test, b30_test = crime_test_lasso.coord_desc(lamb)

    all_variables = df_train.columns[1:]

    nz_train = {all_variables[i]: w30_train[i]
                for i in range(len(w30_train)) if w30_train[i] != 0}
    nz_test = {all_variables[i]: w30_test[i]
               for i in range(len(w30_test)) if w30_test[i] != 0}

    output = open(c.results_path + "a5e" + c.txt_exten, "w")

    output.write("nonzero training weights\n")
    output.write(str(tabulate(nz_train.items())))
    
    output.write("\n\nnonzero test weights\n")
    output.write(str(tabulate(nz_test.items())))

    # find the min/max entries
    output.write("\n\ntraining min/max\n")
    min_train = min(nz_train, key=nz_train.get)
    output.write(min_train + " : " + str(nz_train.get(min_train)) + "\n")
    max_train = max(nz_train, key=nz_train.get)
    output.write(max_train + " : " + str(nz_train.get(max_train)) + "\n")

    output.write("\ntest min/max\n")
    min_test = min(nz_test, key=nz_test.get)
    output.write(min_test + " : " + str(nz_test.get(min_test)) + "\n")
    max_test = max(nz_test, key=nz_test.get)
    output.write(max_test + " : " + str(nz_test.get(max_test)) + "\n")

    output.close()


if __name__ == "__main__":
  main()
