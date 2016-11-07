import functools
import operator
import pickle

import numpy as np

def fildir(x):
    return [n for n in dir(x) if not n.startswith('__')]

def product(x):
  return functools.reduce(operator.mul, x, 1)

def one_hot(num_classes, y):
  d = []
  for i in range(num_classes):
    e = np.zeros(num_classes)
    e[i] = 1
    d.append(e)
  fn = lambda yr: d[yr[0]]
  return np.apply_along_axis(fn, 1, y.reshape((y.shape[0], 1)))

# Undo a one-hot encoding...
def un_hot(num_classes, y):
  fn = lambda yr: np.argmax(yr)
  return np.apply_along_axis(fn, 1, y)

def pickle_to(x, filename):
  with open(filename, 'wb') as f:
    pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle_from(filename):
  with open(filename, 'rb') as f:
    return pickle.load(f)
