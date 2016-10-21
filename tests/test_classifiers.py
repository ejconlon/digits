import sys
for p in ['.']:
  if p not in sys.path:
    sys.path.insert(0, p)

import pytest
from sklearn.metrics import accuracy_score

from digits.common import un_hot
from digits.data import Env, Loader, prepare_cropped
from digits.classifiers import train_and_test_model

env = Env('.')
env.assert_ready()
loader = Loader.from_env(env)
loader.assert_ready()
train_data = prepare_cropped(loader.read_cropped('train'), 1000, gray=True)
test_data = prepare_cropped(loader.read_cropped('test'), 100, gray=True)

def acc(actual, expected):
  assert len(expected.shape) == 1
  assert len(actual.shape) == 2
  assert actual.shape[0] == expected.shape[0]
  num_classes = actual.shape[1]
  return accuracy_score(un_hot(num_classes, actual), expected)

def test_baseline():
  _, train_pred, test_pred = train_and_test_model(env, 'baseline', train_data, test_data)
  print('baseline train acc', acc(train_pred, train_data.y))
  print('baseline test acc', acc(test_pred, test_data.y))

def test_tf():
  _, train_pred, test_pred = train_and_test_model(env, 'tf', train_data, test_data)
  print('tf train acc', acc(train_pred, train_data.y))
  print('tf test acc', acc(test_pred, test_data.y))
