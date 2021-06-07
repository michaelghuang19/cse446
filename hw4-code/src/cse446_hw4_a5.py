# -*- coding: utf-8 -*-
"""cse446_hw4_a5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SINxgwEQvCW2t4oVhfGCk7GH3LvjMkmZ
"""

# Commented out IPython magic to ensure Python compatibility.
# A5
# Imports

# %matplotlib inline

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Constants 

k = 10
k_set = [2, 4, 8, 16, 32, 64]

# Helpers

def plot_single(data, title):
  print("plotting objective functions")

  iterations = list(range(1, len(data) + 1))

  plt.plot(iterations, data)

  plt.title("objective function over time")
  plt.xlabel("iterations")
  plt.ylabel("objective function")

  plt.savefig(title)

def plot_error(data, title):
  print("plotting loss")

  for item in data:
    plt.plot(k_set, item)

  plt.title("error over k")
  plt.xlabel("k")
  plt.ylabel("error")
  plt.legend(["training", "test"])

  plt.savefig(title)

# import MNIST

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                               ]))
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                               ]))

train_loader = DataLoader(train_data, batch_size=len(train_data))
test_loader = DataLoader(test_data, batch_size=len(test_data))

train_data = next(iter(train_loader))[0].numpy()
test_data = next(iter(test_loader))[0].numpy()

def run_lloyd(data, k, function):
  n = data.shape[0]

  # initialize values
  old_idx = np.zeros(n)
  new_idx =  np.zeros(n)

  centroids = data[np.random.choice(n, size=k, replace=False)]

  result_list = []

  iteration = 0
  while True:
    print("iteration " + str(iteration))

    old_idx = np.copy(new_idx)
    new_idx = classify(data, centroids)

    centroids = recenter(data, new_idx, centroids)

    # calculating objective/error  

    if function == "b":
      error = np.sum(np.square(np.linalg.norm(data - centroids[new_idx], axis = data.ndim - 1)))

    if function == "c":
      square = np.square(np.linalg.norm(
          data - centroids[new_idx], axis=data.ndim - 1))
      sum = np.sum(square, axis=1)
      sum = np.sum(sum, axis=1)
      error = np.mean(np.min(sum))

    # if no more changes, then break
    if np.array_equal(old_idx, new_idx):
      break

    iteration +=1
  
  return centroids, result_list, error, iteration

def classify(data, centroids):
  print("classifying points to centroids")

  closest_centroids = []

  for point in data:
    closest_centroids.append(np.argmin([np.linalg.norm(point - centroid) for centroid in centroids]))

  return np.array(closest_centroids)

def recenter(data, closest, centroids):
  print("assigning centroids")
  
  new_centroids = []

  for i, centroid in enumerate(centroids):
    new_centroids.append(data[closest == i].mean(axis=0))

  return np.array(new_centroids)

def get_obj(data, centroids):
  obj = ((np.square(np.linalg.norm(data - centroids[new_idx], axis = data.ndim - 1))).sum() / n)

  return np.array(obj)

centroids, obj_list, _, iteration = run_lloyd(train_data, k, "b")

print(iteration)
print(obj_list)

# plot b
plot_single(obj_list, "a5.png")

# visualize centers

fig, axes = plt.subplots(2, 5)
axes_list = []
for item in axes:
  axes_list += list(item)

for i, ax in enumerate(axes_list):
  ax.imshow(centroids[i][0])

train_error_final = []
test_error_final = []

train_iter_counts = []
test_iter_counts = []

for k_val in k_set:
  _, _, last_train, train_iter = run_lloyd(train_data, k_val, "c")
  _, _, last_test, test_iter = run_lloyd(test_data, k_val, "c")

  train_error_final.append(last_train)
  test_error_final.append(last_test)

  train_iter_counts.append(train_iter)
  test_iter_counts.append(test_iter)

# plot c
print(train_iter_counts)
print(train_error_final)
print(test_iter_counts)
print(test_error_final)

plot_error([train_error_final, test_error_final], "a5_b.png")
