import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits import mplot3d

def get_zvalue(x, y):
  return -x - y

def plot_hyperplanes():
  print("A.9.a")

  x = np.arange(-100, 100)
  y = np.arange(-100, 100)
  plt.title("A.9.a")
  plt.xlabel("x-values")
  plt.ylabel("y-values")
  plt.title("sine wave form")
  plt.xlim(-10, 10)
  plt.ylim(-10, 10)
  plt.plot(x, 0 * x)
  plt.plot(0 * y, y)
  plt.plot(x, 0.5 * x - 1)
  plt.savefig("A9a.png")
  plt.close()

  print("A.9.b")
  fig = plt.figure()
  ax = plt.axes(projection='3d')

  # x and y axis
  x = np.linspace(-80, 80)
  y = np.linspace(-80, 80)
  z = np.linspace(-80, 80)

  xo = np.linspace(-100, 100)
  yo = np.linspace(-100, 100)
  zo = np.linspace(-100, 100)

  X, Y = np.meshgrid(x, y)
  Z = get_zvalue(X, Y)

  ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
  ax.plot3D(xo, yo * 0, zo * 0)
  ax.plot3D(xo * 0, yo, zo * 0)
  ax.plot3D(xo * 0, yo * 0, zo)
  ax.set_title("A.9.b")

  ax.set_xlabel("x-values")
  ax.set_ylabel("y-values")
  ax.set_zlabel("z-values")

  plt.savefig("A9b.png");
  plt.close()

def matrix_compute():
  a = np.array([
    [0, 2, 4],
    [2, 4, 2],
    [3, 3, 1]
  ])

  b = np.array([
    [-2],
    [-2],
    [-4]
  ])

  c = np.array([
    [1],
    [1],
    [1]
  ])
  
  # A.11 a. A^-1
  print("A.11.a")
  ainv = np.linalg.inv(a)
  print(ainv)

  # A.11 b. A^-1b, Ac
  print("A.11.b")
  ainv_b = ainv.dot(b)
  print(ainv_b)
  ac = a.dot(c)
  print(ac)

# Use Z=numpy.random.randn(n) to generate the random variables;
# plt.step(sorted(Z), np.arange(1,n+1)/float(n)) to plot
def plot_rvs():
  print("A.12.a")

  print("A.12.b")


def main():
  plot_hyperplanes()
  matrix_compute()
  plot_rvs()

if __name__ == "__main__":
  main()
