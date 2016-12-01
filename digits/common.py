"""
Various utility functions.
"""

import functools
import operator
import pickle
import sys

import numpy as np

def fildir(x):
  """
  `dir` output filtering out magic attributes.

  Args:
    x (any): anything

  Returns:
    (list[str]) list of non-magic attributes of an object
  """
  return [n for n in dir(x) if not n.startswith('__')]

def product(x):
  """
  Product of all elements in the iterable.

  Args:
    x (iterable): a numeric iterable

  Returns:
    (numeric) product of all elements
  """
  return functools.reduce(operator.mul, x, 1)

def one_hot(num_classes, y):
  """
  Simple one-hot encoder.

  Args:
    num_classes (int): number of classes
    y (ndarray (n,)): labels

  Returns:
    (ndarray (n, num_classes)) one-hotted labels
  """
  d = []
  for i in range(num_classes):
    e = np.zeros(num_classes)
    e[i] = 1
    d.append(e)
  fn = lambda yr: d[yr[0]]
  return np.apply_along_axis(fn, 1, y.reshape((y.shape[0], 1)))

def un_hot(num_classes, y):
  """
  Undo a one-hot encoding by argmaxing.

  Args:
    num_classes (int): number of classes
    y (ndarray (n, num_classes)): one-hot labels

  Returns:
    (ndarray (n,)) un-one-hotted labels
  """
  fn = lambda yr: np.argmax(yr)
  return np.apply_along_axis(fn, 1, y)

class FileWrap:
  """
  OSX Python3 has a bug that prevents reading/writing files over 2GB
  so this just buffers reads and writes. Pretend this doesn't exist.
  """

  def __init__(self, f):
    self.f = f
  def write(self, bytes):
    lim = (1 << 31) - 1
    if len(bytes) > lim:
      self.f.write(bytes[:lim])
      self.f.write(bytes[lim:])
    else:
      self.f.write(bytes)
  def read(self, size=None):
    bytes = bytearray()
    lim = (1 << 31) - 1
    while size > 0:
      chunk = self.f.read(min(size, lim))
      size -= len(chunk)
      bytes.extend(chunk)
      if len(chunk) == 0:
        break
    return bytes
  def readline(self, size=None):
    raise Exception("Not supported")

def pickle_to(x, filename):
  """
  Wraps pickle.dump (see FileWrap)

  Args:
    x (any): obj to pickle
    filename (str): file to pickle to
  """
  with open(filename, 'wb') as f:
    if sys.version_info > (3,):
      g = FileWrap(f)
      pickle.dump(x, g, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)

def unpickle_from(filename):
  """
  Wraps pickle.load (see FileWrap)

  Args:
    filename (str): file to unpickle from

  Returns:
    (any) unpickled object
  """
  with open(filename, 'rb') as f:
    if sys.version_info > (3,):
      g = FileWrap(f)
      return pickle.load(g)
    else:
      return pickle.load(f)
