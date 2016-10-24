from collections import namedtuple
import os
import pickle

import numpy as np
from scipy.io import loadmat
from skimage.color import rgb2gray

from .common import product

Data = namedtuple('Data', ['X', 'y', 'offset', 'inv_map'])

class Env:
  def __init__(self, path):
    self.path = os.path.abspath(path)

  def __getattr__(self, name):
    attr_path = os.path.join(self.path, name)
    assert os.path.isdir(attr_path)
    return attr_path

  def assert_ready(self):
    assert os.path.isdir(self.path)

class Loader:
  @classmethod
  def from_env(cls, env):
    return Loader(env.data, env.pickled)

  def __init__(self, data_path, pickled_path):
    self.data_path = data_path
    self.pickled_path = pickled_path
    self.mat_suffix = '_32x32.mat'
    self.pickle_suffix = '.pickle'
      
  def assert_ready(self):
    assert os.path.isdir(self.data_path)
    assert os.path.isdir(self.pickled_path)
      
  def clear_pickled(self):
    files = os.listdir(self.pickled_path)
    for file in files:
      if file.endswith('.pickle'):
        os.remove(os.path.join(self.pickled_path, file))
      
  # TODO support more than cropped
  def read_cropped(self, role):
    mat_file = os.path.join(self.data_path, role + self.mat_suffix)
    pickle_file = os.path.join(self.pickled_path, role + self.mat_suffix + self.pickle_suffix)
    if not os.path.isfile(pickle_file):
      mat = loadmat(mat_file)
      assert len(mat['X'].shape) == 4
      assert len(mat['y'].shape) == 2
      assert mat['X'].shape[-1] == mat['y'].shape[0]
      assert mat['y'].shape[1] == 1
      X = mat['X']
      y = mat['y'].ravel().astype(np.int32)
      assert len(y.shape) == 1
      # Cleanup X: make row-oriented :(
      X = np.moveaxis(X, 3, 0)
      assert X.shape[0] == y.shape[0]
      # Cleanup y: '10' really means '0' :(
      y = np.vectorize(lambda i: 0 if i == 10 else i)(y)
      data = Data(X=X, y=y, offset=0, inv_map=None)
      with open(pickle_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
      return data
    else:
      with open(pickle_file, 'rb') as f:
        return pickle.load(f)
        
  def raw_pickle_file(self, name):
    return os.path.join(self.pickled_path, name + self.pickle_suffix)
      
  def write_raw(self, name, raw):
    with open(self.raw_pickle_file(name), 'wb') as f:
      pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)
      
  def raw_exists(self, name):
    return os.path.isfile(self.raw_pickle_file(name))
  
  def del_raw(self, name):
    os.remove(self.raw_pickle_file(name))
      
  def read_raw(self, name):
    with open(self.raw_pickle_file(name), 'rb') as f:
      return pickle.load(f)

  def load_data(self, name):
    pc = lambda d: prepare_cropped(pc, keep=k, gray=True, shuffle=True)
    if name == 'crop-train':
      return prepare_cropped(self.read_cropped('train'), gray=True, shuffle=True, then_keep=1000)
    elif name == 'crop-valid' or name == 'crop-test':
      return prepare_cropped(self.read_cropped('test'), gray=True, shuffle=True, then_keep=100)
    else:
      raise Exception('Unknown dataset: ' + name)


# TODO consider distribution across all classes
def prepare_cropped(data, drop=None, keep=None, gray=False, shuffle=False, then_keep=None):
  assert data.X.shape[0] == data.y.shape[0]
  assert data.offset == 0
  assert data.inv_map is None
  X = data.X
  y = data.y
  if drop is not None:
    X = X[drop:]
    y = y[drop:]
  if keep is not None:
    X = X[:keep]
    y = y[:keep]
    lim = keep
  else:
    lim = len(X)
  if gray:
    X = rgb2gray(X)
  X = X.astype(np.float32)
  X = X.reshape((X.shape[0], product(X.shape[1:])))
  if shuffle:
    inv_map = np.arange(lim, dtype=np.int32)
    np.random.shuffle(inv_map)
    X = X[inv_map]
    y = y[inv_map]
  else:
    inv_map = None
  if then_keep is not None:
    X = X[:then_keep]
    y = y[:then_keep]
    if inv_map is not None:
      inv_map = inv_map[:then_keep]
  return Data(X=X, y=y, offset=drop, inv_map=inv_map)
