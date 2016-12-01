"""
Tests for each type of classifier.
If these execute successfully, `make train` will too.
"""

import sys
for p in ['.']:
  if p not in sys.path:
    sys.path.insert(0, p)

from argparse import Namespace
import random

import pytest

from digits.data import Env, Loader
from digits.main import sub_main

env = Env('.')
env.assert_ready()
loader = Loader.from_env(env)
loader.assert_ready()

random_state = 71

def test_baseline_crop():
  args = Namespace(
    op='drive',
    model='baseline',
    variant='crop-small',
    random_state=random_state,
    train=True,
    max_acc=None
  )
  sub_main(env, loader, args)

def test_mnist():
  args = Namespace(
    op='drive',
    model='tf',
    variant='mnist',
    random_state=random_state,
    train=True,
    max_acc=0.8
  )
  sub_main(env, loader, args)

def test_crop():
  args = Namespace(
    op='drive',
    model='tf',
    variant='crop-small',
    random_state=random_state,
    train=True,
    max_acc=0.4
  )
  sub_main(env, loader, args)
