import sys
for p in ['.']:
  if p not in sys.path:
    sys.path.insert(0, p)

import pytest

from digits.data import *
from digits.classifiers import *

env = Env('.')
env.assert_ready()
loader = Loader.from_env(env)
loader.assert_ready()
train_data = prepare_cropped(loader.read_cropped('train'), 1000, gray=True)
test_data = prepare_cropped(loader.read_cropped('test'), 100, gray=True)

def test_baseline():
  train_acc, test_acc = run_baseline(train_data, test_data)
  print("Baseline train acc", train_acc)
  print("Baseline test acc", test_acc)

def test_tf():
  train_acc, test_acc = run_tf(train_data, test_data)
  print("TF train acc", train_acc)
  print("TF test acc", test_acc)
