import sys
for p in ['.']:
  if p not in sys.path:
    sys.path.insert(0, p)

from argparse import Namespace

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from digits.common import un_hot
from digits.classifiers import run_train_model, run_test_model
from digits.data import Env, Loader, prepare_cropped
from digits.metrics import Metrics
from digits.main import write_results

env = Env('.')
env.assert_ready()
loader = Loader.from_env(env)
loader.assert_ready()
orig = loader.read_cropped('test')
train_data = prepare_cropped(orig, keep=1000, gray=True)
valid_data = prepare_cropped(orig, drop=1000, keep=100, gray=True)

def acc(actual, expected):
  assert len(expected.shape) == 1
  assert len(actual.shape) == 2
  assert actual.shape[0] == expected.shape[0]
  num_classes = actual.shape[1]
  return accuracy_score(un_hot(num_classes, actual), expected)

def run_model(model):
  train_pred, valid_pred = run_train_model(env, model, None, train_data, valid_data)
  valid_pred2 = run_test_model(env, model, None, valid_data)
  np.testing.assert_array_equal(valid_pred, valid_pred2)
  args = Namespace(model=model, variant=None)
  write_results(env, args, 'test', orig, valid_data, valid_pred)

def test_baseline():
  run_model('baseline')

def test_tf():
  run_model('tf')
