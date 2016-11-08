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
from digits.main import sub_main

env = Env('.')
env.assert_ready()
loader = Loader.from_env(env)
loader.assert_ready()

# random_state = random.randint(0, 1000)
random_state = 70

def acc(actual, expected):
  assert len(expected.shape) == 1
  assert len(actual.shape) == 2
  assert actual.shape[0] == expected.shape[0]
  num_classes = actual.shape[1]
  return accuracy_score(un_hot(num_classes, actual), expected)

def run_model(
  model, variant, train_data_name, valid_data_name, test_data_name,
  preprocessor, param_set, search_set=None, search_size=None, check_ser=False):
  train_args = Namespace(
    random_state=random_state,
    op='train',
    model=model,
    variant=variant,
    train_data=train_data_name,
    valid_data=valid_data_name,
    test_data=test_data_name,
    preprocessor=preprocessor,
    param_set=param_set,
    search_set=search_set,
    search_size=search_size
  )
  sub_main(env, loader, train_args)

  if check_ser and test_data_name is not None:
    test_metrics1 = unpickle_from(env.resolve_role_file(model, variant, 'test', 'metrics.pickle'))

    test_args = Namespace(
      random_state=random_state,
      op='test',
      model=model,
      variant=variant,
      test_data=test_data_name,
      preprocessor=preprocessor,
      param_set=param_set
    )
    sub_main(env, loader, test_args)

    test_metrics2 = unpickle_from(env.resolve_role_file(model, variant, 'test', 'metrics.pickle'))

    # Sanity check, they should have run on the same dataset
    np.testing.assert_array_equal(test_metrics1.gold, test_metrics2.gold)

    # Now check that we've correctly predicted with the serialized model
    np.testing.assert_array_equal(test_metrics1.pred, test_metrics2.pred)

# def test_baseline_crop():
#   run_model(
#     model='baseline',
#     variant='crop',
#     train_data_name='crop-train-small',
#     valid_data_name=None,
#     test_data_name='crop-test-small',
#     preprocessor='flat-gray',
#     param_set='crop'
#   )

# def test_baseline_mnist():
#   run_model(
#     model='baseline',
#     variant='mnist',
#     train_data_name='mnist-train',
#     valid_data_name=None,
#     test_data_name='mnist-test',
#     preprocessor='flat-gray',
#     param_set='mnist'
#   )

# def test_tf_crop_huge():
#   run_model(
#     model='tf',
#     variant='crop-huge',
#     train_data_name='crop-train-huge',
#     valid_data_name=None,
#     test_data_name='crop-test-huge',
#     preprocessor='color',
#     param_set='crop'
#   )

# def test_tf_crop_big():
#   run_model(
#     model='tf',
#     variant='crop-big',
#     train_data_name='crop-train-big',
#     valid_data_name=None,
#     test_data_name='crop-test-big',
#     preprocessor='color',
#     param_set='crop'
#   )

# def test_tf_crop_small():
#   run_model(
#     model='tf',
#     variant='crop-small',
#     train_data_name='crop-train-small',
#     valid_data_name=None,
#     test_data_name='crop-test-small',
#     preprocessor='color',
#     param_set='crop'
#   )

def test_tf_mnist():
  run_model(
    model='tf',
    variant='mnist',
    train_data_name='mnist-train',
    valid_data_name=None,
    test_data_name='mnist-test',
    preprocessor='color',
    param_set='mnist',
    check_ser=True
  )
