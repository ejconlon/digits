import numpy as np
from sklearn.metrics import accuracy_score

class Metrics:
  def __init__(self, num_classes, orig, inv_map, actual_hot, expected):
    self.num_classes = num_classes
    self.orig = orig
    self.inv_map = inv_map
    self.actual_hot = actual_hot
    self.actual = un_hot(num_classes, actual_hot)
    self.expected = expected

  def accuracy(self):
    return accuracy_score(self.actual, self.expected)

  def entropy(self):
    raise Exception("TODO")

