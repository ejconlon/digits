import functools
import operator

def fildir(x):
    return filter(lambda n: not n.startswith('__'), dir(x))

def product(x):
  return functools.reduce(operator.mul, x, 1)
