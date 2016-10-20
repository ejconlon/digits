import sys
for p in ['.', '..']:
  if p not in sys.path:
    sys.path.insert(0, p)

import pytest
import numpy as np
from numpy.testing import assert_allclose
from digits.whatever import *

def test_something():
  assert something == 1
