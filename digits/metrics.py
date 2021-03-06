"""
Metrics to evaluate classifier performance.
"""

from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
import json_tricks.np

from .common import un_hot, unpickle_from, pickle_to

# Various metrics
Report = namedtuple('Report', [
  'num_classes',     # number of classes
  'num_samples',     # number of examples
  'accuracy',        # total accuracy
  'precision',       # per-class precision
  'recall',          # per-class recall
  'f1',              # per-class F1
  'confusion',       # class-class confusion
  'gold_class_dist', # distribution of actual labels
  'pred_class_dist'  # distribution of predicted labels
])

# Interesting example DataFrames
Viz = namedtuple('Viz', [
  'correct_certain',   # things we got right confidently
  'wrong_certain',     # things we got wrong confidently
  'correct_uncertain', # things we got right weakly
  'wrong_uncertain'    # things we got wrong weakly
])

def write_report(report, filename):
  """
  Write a report to disk as special JSON.

  Args:
    report (Report) the report
    filename (str) path to output file
  """
  with open(filename, 'w') as f:
    json_tricks.np.dump(report._asdict(), f, sort_keys=True, indent=2)

def read_report(filename):
  """
  Read a report from disk as special JSON.

  Args:
    filename (str) path to input file
  """
  with open(filename, 'r') as f:
    return Report(**json_tricks.np.load(f))

class Metrics:
  """
  Contains predicted and actual labels. Generates interesting metric values.

  Attributes:
    num_classes (int) number of classes
    pred_hot (ndarray) (n, num_classes) array of prediction probabilities
    gold (ndarray) (n,) array of correct labels
  """

  def __init__(self, num_classes, pred_hot, gold):
    assert len(gold.shape) == 1
    assert len(pred_hot.shape) == 2
    assert pred_hot.shape[0] == gold.shape[0]
    assert pred_hot.shape[1] == num_classes
    self.num_classes = num_classes
    self.pred_hot = pred_hot
    self.pred = un_hot(num_classes, pred_hot)
    self.gold = gold

  def num_samples(self):
    """
    Returns: (int) number of examples
    """
    return self.gold.shape[0]

  def accuracy(self):
    """
    Returns: (float) accuracy
    """
    return sklearn.metrics.accuracy_score(self.gold, self.pred)

  def prfs(self):
    """
    Returns: (precision, recall, fscore, support)
    """
    return sklearn.metrics.precision_recall_fscore_support(self.gold, self.pred, labels=range(self.num_classes), warn_for=())

  def confusion(self):
    """
    Returns: (ndarray) confusion matrix
    """
    return sklearn.metrics.confusion_matrix(self.gold, self.pred)

  def gold_class_dist(self):
    """
    Returns: (ndarray) gold class distribution
    """
    return np.histogram(self.gold, bins=self.num_classes, density=True)[0]

  def pred_class_dist(self):
    """
    Returns: (ndarray) predicted class distribution
    """
    return np.histogram(self.pred, bins=self.num_classes, density=True)[0]

  def entropy(self):
    """
    Returns: (ndarray) entropy each example's prediction distibution
    """
    e = np.apply_along_axis(scipy.stats.entropy, 1, self.pred_hot)
    assert len(e.shape) == 1
    assert e.shape[0] == self.pred_hot.shape[0]
    return e

  def correct_indices(self):
    """
    Returns: (ndarray) indices for examples we got right
    """
    return np.where(self.pred == self.gold)[0]

  def wrong_indices(self):
    """
    Returns: (ndarray) indices for examples we got wrong
    """
    return np.where(self.pred != self.gold)[0]

  def most_uncertain_indices(self, e=None):
    """
    Returns: (ndarray) indices for examples we were least sure of
    """
    return list(reversed(self.most_certain_indices(e)))

  def most_certain_indices(self, e=None):
    """
    Returns: (ndarray) indices for examples we were most sure of
    """
    if e is None:
      e = self.entropy()
    c = np.argsort(e)
    assert c.shape == e.shape
    return c

  def report(self):
    """
    Returns: (Report) report of all these metrics
    """
    precision, recall, f1, support = self.prfs()
    return Report(
      num_classes = self.num_classes,
      num_samples = self.num_samples(),
      accuracy = self.accuracy(),
      precision = precision,
      recall = recall,
      f1 = f1,
      confusion = self.confusion(),
      gold_class_dist = self.gold_class_dist(),
      pred_class_dist = self.pred_class_dist()
    )

  def print_classification_report(self):
    """
    Prints the report to stdout.
    """
    print(sklearn.metrics.classification_report(self.gold, self.pred))

  def viz(self, proc, k):
    """
    Finds interesting examples. (most/least certain wrong/right)

    Args:
      proc (Data): original data
      k (int): number of examples to save

    Returns:
      (Viz) interesting examples
    """
    correct = self.correct_indices()
    correct_set = set(correct)
    entropy = self.entropy()
    certain = self.most_certain_indices(entropy)
    uncertain = self.most_uncertain_indices(entropy)
    indices = {
      'correct_certain': [i for i in certain if i in correct_set][:k],
      'wrong_certain': [i for i in certain if i not in correct_set][:k],
      'correct_uncertain': [i for i in uncertain if i in correct_set][:k],
      'wrong_uncertain': [i for i in uncertain if i not in correct_set][:k]
    }
    inv_map = proc.inv_map
    if inv_map is None:
      inv_map = list(range(proc.offset, proc.offset + len(proc.X)))
    sets = {}
    columns = ['index', 'gold_class', 'pred_class', 'p', 'entropy', 'proc_image']
    for k, v in indices.items():
      recs = []
      for i in v:
        gold_class = self.gold[i]
        assert proc.y[i] == gold_class
        pred_class = self.pred[i]
        p = self.pred_hot[i][pred_class]
        rec = {
          'index': i,
          'gold_class': gold_class,
          'pred_class': pred_class,
          'p': p,
          'entropy': entropy[i],
          'proc_image': proc.X[i]
        }
        recs.append(rec)
      sets[k] = pd.DataFrame.from_records(recs, columns=columns)
    return Viz(**sets)
