# graph.py
# HW3 A6 
# Applicable raw data graphs since Colab was being difficult

import matplotlib.pyplot as plt
import numpy as np

import data as d
import helpers as h

graph_a = True
graph_b = False
graph_c = False
graph_d = False

'''
Arbitrarily section different datasets
'''


def graph():
  if graph_a:
    print("graphing a")
    
    a_groupings = [[0, 4], [4, 8], [8, 12]]
    
    for idx, grouping in enumerate(a_groupings):
      start = grouping[0]
      end = grouping[1]

      h.plot_acc(d.a_train[start:end], d.a_labels[start:end],
                 "a6_a_t{}.png".format(idx), "training")
      h.plot_acc(d.a_valid[start:end], d.a_labels[start:end],
                 "a6_a_v{}.png".format(idx), "validation")

    best_idx = 10
    h.plot_acc([d.a_train[best_idx]], [d.a_labels[best_idx]],
               "a6_a_tb.png", "training best hyperparameters")
    h.plot_acc([d.a_valid[best_idx]], [d.a_labels[best_idx]],
               "a6_a_vb.png", "validation best hyperparameters")

  if graph_b:
    print("graphing b")

    

  if graph_c:
    print("graphing c")

  if graph_d:
    print("graphing d")



graph()
