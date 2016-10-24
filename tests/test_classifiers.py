import sys
for p in ['.']:
  if p not in sys.path:
    sys.path.insert(0, p)

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from digits.common import un_hot
from digits.classifiers import train_and_test_model
from digits.data import Env, Loader, prepare_cropped
from digits.metrics import Metrics

env = Env('.')
env.assert_ready()
loader = Loader.from_env(env)
loader.assert_ready()
test = loader.read_cropped('test')
train_data = prepare_cropped(test, keep=1000, gray=True)
valid_data = prepare_cropped(test, drop=1000, keep=100, gray=True)

def acc(actual, expected):
  assert len(expected.shape) == 1
  assert len(actual.shape) == 2
  assert actual.shape[0] == expected.shape[0]
  num_classes = actual.shape[1]
  return accuracy_score(un_hot(num_classes, actual), expected)

def run_model(model):
  train_pred, valid_pred, valid_pred2 = train_and_test_model(env, model, None, train_data, valid_data, valid_data)
  np.testing.assert_array_equal(valid_pred, valid_pred2)
  metrics = Metrics(10, valid_pred, valid_data.y)
  report = metrics.report()
  metrics.print_classification_report()

def test_baseline():
  run_model('baseline')

def test_tf():
  run_model('tf')
