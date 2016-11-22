import functools
import operator
import pickle
import sys

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

# OSX Python has a bug that prevents reading/writing files over 2GB
# so this just buffers reads and writes
class FileWrap:
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
  with open(filename, 'wb') as f:
    if sys.version_info > (3,):
      g = FileWrap(f)
      pickle.dump(x, g, protocol=pickle.HIGHEST_PROTOCOL)
    else:
      pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_from(filename):
  with open(filename, 'rb') as f:
    if sys.version_info > (3,):
      g = FileWrap(f)
      return pickle.load(g)
    else:
      return pickle.load(f)
