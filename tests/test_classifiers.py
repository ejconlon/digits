import sys
for p in ['.']:
  if p not in sys.path:
    sys.path.insert(0, p)

from argparse import Namespace
import random

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from digits.common import un_hot
from digits.data import Env, Loader
from digits.metrics import Metrics, unpickle_from
from digits.main import run_test, run_train

env = Env('.')
env.assert_ready()
loader = Loader.from_env(env)
loader.assert_ready()

random_state = random.randint(0, 1000)

def acc(actual, expected):
  assert len(expected.shape) == 1
  assert len(actual.shape) == 2
  assert actual.shape[0] == expected.shape[0]
  num_classes = actual.shape[1]
  return accuracy_score(un_hot(num_classes, actual), expected)

def run_model(model, variant, train_data_name, test_data_name):
  train_args = Namespace(model=model, variant=variant, train_data=train_data_name, valid_data=test_data_name, test_data=None, random_state=random_state)
  run_train(env, loader, train_args)

  test_args = Namespace(model=model, variant=variant, test_data=test_data_name, random_state=random_state)
  run_test(env, loader, test_args)

  valid_metrics = unpickle_from(env.resolve_role_file(model, variant, 'valid', 'metrics.pickle'))
  test_metrics = unpickle_from(env.resolve_role_file(model, variant, 'test', 'metrics.pickle'))

  # Sanity check, they should have run on the same dataset
  np.testing.assert_array_equal(valid_metrics.gold, test_metrics.gold)

  # Now check that we've correctly predicted with the serialized model
  np.testing.assert_array_equal(valid_metrics.pred, test_metrics.pred)

def test_baseline():
  run_model('baseline', 'testing', 'crop-train-small', 'crop-test-small')

def test_tf():
  run_model('tf', 'testing', 'crop-train-small', 'crop-test-small')
