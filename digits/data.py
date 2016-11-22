from collections import namedtuple
import os
import pickle
import random
import shutil

import numpy as np
from scipy.io import loadmat

from .common import product, unpickle_from, pickle_to
from .preprocessors import PREPROCESSORS

Data = namedtuple('Data', ['X', 'y', 'offset', 'inv_map'])

class Env:
  def __init__(self, path):
    self.path = os.path.abspath(path)

  def resolve(self, name):
    attr_path = os.path.join(self.path, name)
    assert os.path.isdir(attr_path)
    return attr_path

  def assert_ready(self):
    assert os.path.isdir(self.path)

  def _mkdir(self, p, clean):
    if not os.path.isdir(p):
      os.makedirs(p)
    elif clean:
      shutil.rmtree(p)
      os.makedirs(p)
    return p

  def _touch(self, p, clean):
    if os.path.isfile(p) and clean:
      os.remove(p)
    return p

  def model_name_plus(self, name, variant):
    return name if variant is None else name + '_' + variant

  def resolve_model(self, name, variant, clean=False):
    assert '.' not in name and (variant is None or '.' not in variant)
    logs_path = self.resolve('logs')
    return self._mkdir(os.path.join(logs_path, self.model_name_plus(name, variant)), clean)
    
  def resolve_role(self, name, variant, role, clean=False):
    model_path = self.resolve_model(name, variant)
    return self._mkdir(os.path.join(model_path, role), clean)

  def resolve_model_file(self, name, variant, filename, clean=False):
    model_path = self.resolve_model(name, variant)
    return self._touch(os.path.join(model_path, filename), clean)

  def resolve_role_file(self, name, variant, role, filename, clean=False):
    role_path = self.resolve_role(name, variant, role)
    return self._touch(os.path.join(role_path, filename), clean)

  def prepare(self, name, variant, roles, remove=False):
    name_path = self.resolve_name_path(name, variant)
    # per-role output dirs
    paths = dict((role, os.path.join(name_path, role)) for role in roles)
    for path in paths.values():
      if os.path.isdir(path) and remove:
        shutil.rmtree(path)
      os.makedirs(path)
    # train artifact files
    artifacts = ['ckpt', 'clf']
    artifact_paths = dict((art, os.path.join(name_path, 'model.' + art)) for art in artifacts)
    if 'train' in roles and remove:
      for path in artifact_paths.values():
        if os.path.isfile(path):
          os.remove(path)
    paths.update(artifact_paths)
    return paths

class Loader:
  @classmethod
  def from_env(cls, env):
    return Loader(env.resolve('data'), env.resolve('pickled'))

  def __init__(self, data_path, pickled_path):
    self.data_path = data_path
    self.pickled_path = pickled_path
      
  def assert_ready(self):
    assert os.path.isdir(self.data_path)
    assert os.path.isdir(self.pickled_path)
      
  def clear_pickled(self):
    files = os.listdir(self.pickled_path)
    for file in files:
      if file.endswith('.pickle'):
        os.remove(os.path.join(self.pickled_path, file))
      
  def read_cropped(self, role):
    mat_file = os.path.join(self.data_path, role + '_32x32.mat')
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
    return Data(X=X, y=y, offset=0, inv_map=None)

  def read_mnist(self):
    mat_file = os.path.join(self.data_path, 'mldata/mnist-original.mat')
    mnist = loadmat(mat_file)
    num = mnist['label'].shape[1]
    y = mnist['label'].reshape((num,)).astype(np.int32)
    X = np.moveaxis(mnist['data'], 1, 0).reshape((num, 28, 28))
    return Data(X=X, y=y, offset=0, inv_map=None)

  def load_data(self, name, preprocessor, random_state):
    """ Return proc """
    print('loading', name, preprocessor, random_state)
    proc_file = os.path.join(self.pickled_path, '.'.join([name, str(random_state), preprocessor, 'proc', 'pickle']))
    if os.path.isfile(proc_file):
      print('unpickling data')
      proc = unpickle_from(proc_file)
      return proc
    else:
      print('deriving data')
    if name == 'crop-train-small':
      orig = self.read_cropped('train')
      proc = prepare_cropped(orig, shuffle=True, then_keep=2000, random_state=random_state)
    elif name == 'crop-valid-small':
      orig = self.read_cropped('train')
      proc = prepare_cropped(orig, shuffle=True, then_drop=2000, then_keep=400, random_state=random_state)
    elif name == 'crop-test-small':
      orig = self.read_cropped('test')
      proc = prepare_cropped(orig, shuffle=True, then_keep=400, random_state=random_state)
    elif name == 'crop-train-big':
      orig = self.read_cropped('train')
      assert orig.X.shape[0] == 73257
      proc = prepare_cropped(orig, shuffle=True, then_keep=60000, random_state=random_state)
    elif name == 'crop-valid-big':
      orig = self.read_cropped('train')
      proc = prepare_cropped(orig, shuffle=True, then_drop=60000, random_state=random_state)
    elif name == 'crop-test-big':
      orig = self.read_cropped('test')
      proc = prepare_cropped(orig, shuffle=True, random_state=random_state)
    elif name == 'crop-train-huge':
      orig0 = self.read_cropped('extra')
      orig1 = self.read_cropped('train')
      orig = concat(orig0, orig1)
      assert orig0.X.shape[0] == 531131
      assert orig1.X.shape[0] == 73257
      assert orig.X.shape[0] == 604388
      assert orig.y.shape[0] == 604388
      del orig0
      del orig1
      proc = prepare_cropped(orig, shuffle=True, then_keep=480000, random_state=random_state)
    elif name == 'crop-valid-huge':
      orig0 = self.read_cropped('extra')
      orig1 = self.read_cropped('train')
      orig = concat(orig0, orig1)
      del orig0
      del orig1
      proc = prepare_cropped(orig, shuffle=True, then_drop=480000, random_state=random_state)
    elif name == 'mnist-train':
      orig = self.read_mnist()
      assert orig.X.shape[0] == 70000
      proc = prepare_cropped(orig, shuffle=True, then_keep=42000, random_state=random_state)
    elif name == 'mnist-valid':
      orig = self.read_mnist()
      proc = prepare_cropped(orig, shuffle=True, then_drop=42000, then_keep=14000, random_state=random_state)
    elif name == 'mnist-test':
      orig = self.read_mnist()
      proc = prepare_cropped(orig, shuffle=True, then_drop=56000, then_keep=14000, random_state=random_state)
    else:
      raise Exception('Unknown dataset: ' + name)
    del orig
    if preprocessor is not None:
      print('preprocessing with', preprocessor)
      proc = PREPROCESSORS[preprocessor](proc)
    pickle_to(proc, proc_file)
    return proc


class RandomStateContext:
  def __init__(self, seed):
    if seed is not None:
      assert isinstance(seed, int)
    self.seed = seed
    self.old_stdlib_state = None
    self.old_numpy_state = None
  def __enter__(self):
    if self.seed is not None:
      self.old_stdlib_state = random.getstate()
      random.seed(self.seed)
      self.old_numpy_state = np.random.get_state()
      new_numpy_state = np.random.RandomState(self.seed)
      np.random.set_state(new_numpy_state.get_state())
  def __exit__(self, *args):
    if self.old_stdlib_state is not None:
      random.setstate(self.old_stdlib_state)
      self.old_stdlib_state = None
    if self.old_numpy_state is not None:
      np.random.set_state(self.old_numpy_state)
      self.old_numpy_state = None

def concat(data0, data1):
  assert data0.X.shape[0] == data0.y.shape[0]
  assert data1.X.shape[0] == data1.y.shape[0]
  assert data0.offset == 0
  assert data1.offset == 0
  assert data0.inv_map is None
  assert data1.inv_map is None
  return Data(X = np.concatenate((data0.X, data1.X)), y = np.concatenate((data0.y, data1.y)), offset = 0, inv_map = None)

def invert(num_classes, y):
  assert len(y.shape) == 1
  inv = [[] for i in range(num_classes)]
  for i in range(y.shape[0]):
    v = y[i]
    inv[v].append(i)
  return [np.array(inv[i], dtype=np.int32) for i in range(num_classes)]

# TODO consider distribution across all classes
# also just consider delegating to train_test_split or whatever
def prepare_cropped(data, drop=None, keep=None, shuffle=False, then_drop=None, then_keep=None, random_state=None):
  assert data.X.shape[0] == data.y.shape[0]
  assert data.offset == 0
  assert data.inv_map is None
  X = data.X
  y = data.y
  offset = 0
  if drop is not None:
    offset = drop
    X = X[drop:]
    y = y[drop:]
  if keep is not None:
    X = X[:keep]
    y = y[:keep]
    lim = keep
  else:
    lim = len(X)
  if shuffle:
    inv_map = np.arange(lim, dtype=np.int32)
    with RandomStateContext(random_state):
      np.random.shuffle(inv_map)
    X = X[inv_map]
    y = y[inv_map]
  else:
    inv_map = None
  if then_drop is not None:
    X = X[then_drop:]
    y = y[then_drop:]
    if inv_map is None:
      offset += then_drop
    else:
      inv_map = inv_map[then_drop:]
  if then_keep is not None:
    X = X[:then_keep]
    y = y[:then_keep]
    if inv_map is not None:
      inv_map = inv_map[:then_keep]
  return Data(X=X, y=y, offset=offset, inv_map=inv_map)
