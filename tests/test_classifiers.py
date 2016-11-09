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

# random_state = random.randint(0, 1000)
random_state = 70

def test_whatever():
  args = Namespace(
    op='drive',
    model='tf',
    variant='mnist',
    random_state=random_state
  )
  sub_main(env, loader, args)
