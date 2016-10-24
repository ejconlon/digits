import sys
for p in ['.']:
  if p not in sys.path:
    sys.path.insert(0, p)

import pytest
import numpy as np

from digits.data import *
from digits.common import *

env = Env('.')
env.assert_ready()
loader = Loader.from_env(env)
loader.assert_ready()

def test_load_test():
  data = loader.read_cropped('test')
  reg = prepare_cropped(data, drop=42, keep=10)
  shuf = prepare_cropped(data, drop=42, keep=10, shuffle=True)
  assert reg.offset == 42
  assert shuf.offset == 42
  assert reg.X.shape == shuf.X.shape
  assert reg.y.shape == shuf.y.shape
  assert reg.inv_map is None
  assert shuf.inv_map is not None
  assert len(shuf.inv_map.shape) == 1
  assert shuf.inv_map.shape[0] == shuf.X.shape[0]
  si = 5
  ri = shuf.inv_map[si]
  xs = shuf.X[si]
  xr = reg.X[ri]
  ys = shuf.y[si]
  yr = reg.y[ri]
  np.testing.assert_array_equal(xs, xr)
  assert ys == yr
  yo = data.y[42 + ri]
  assert yo == yr

def test_load_raw():
  name = 'foo_bar_baz'
  if loader.raw_exists(name):
    loader.del_raw(name)
  assert not loader.raw_exists(name)
  raw = {'a': 1, 'b': 2}
  loader.write_raw(name, raw)
  assert loader.raw_exists(name)
  raw_again = loader.read_raw(name)
  assert raw_again == raw
  loader.del_raw(name)
  assert not loader.raw_exists(name)

def test_hot():
  num_classes = 5
  r = np.array([2, 0, 4])
  r_hot = one_hot(num_classes, r)
  expected = \
    np.array([
      [0, 0, 1, 0, 0],
      [1, 0, 0, 0, 0],
      [0, 0, 0, 0, 1]
    ])
  np.testing.assert_equal(expected, r_hot)
  r_again = un_hot(5, r_hot)
  np.testing.assert_equal(r_again, r)

