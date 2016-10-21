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

def test_load_raw():
  loader = Loader.from_env(env)
  loader.assert_ready()
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

