import sys
for p in ['.']:
  if p not in sys.path:
    sys.path.insert(0, p)

import pytest
import numpy as np

from digits.data import *

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
