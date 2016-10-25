from abc import ABCMeta, abstractmethod
from collections import namedtuple
import os
import pickle
import shutil

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf

from .common import one_hot
from .metrics import Metrics, write_report, read_report

class Model(metaclass=ABCMeta):
  def __init__(self, env, name, variant, num_features, num_classes):
    self.env = env
    self.name = name
    self.variant = variant
    self.num_features = num_features
    self.num_classes = num_classes

  def _resolve_model(self, clean=False):
    return self.env.resolve_model(self.name, self.variant, clean)

  def _resolve_role(self, role, clean=False):
    return self.env.resolve_role(self.name, self.variant, role, clean)

  def _resolve_model_file(self, filename, clean=False):
    return self.env.resolve_model_file(self.name, self.variant, filename, clean)

  def _resolve_role_file(self, role, filename, clean=False):
    return self.env.resolve_role_file(self.name, self.variant, role, filename, clean)

  @abstractmethod
  def train(self, train_data, valid_data):
    """ Return (one_hot preds, one_hot preds) """
    pass

  @abstractmethod
  def test(self, test_data):
    """ Return (one_hot preds) """
    pass

class BaselineModel(Model):
  def train(self, train_data, valid_data):
    clf_file = self._resolve_model_file('model.clf', clean=True)
    clf = LogisticRegression()
    clf.fit(train_data.X, train_data.y)
    with open(clf_file, 'wb') as f:
      pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    train_pred = clf.predict(train_data.X)
    valid_pred = clf.predict(valid_data.X)
    return (one_hot(self.num_classes, train_pred), one_hot(self.num_classes, valid_pred))

  def test(self, test_data):
    clf_file = self._resolve_model_file('model.clf')
    with open(clf_file, 'rb') as f:
      clf = pickle.load(f)
    test_pred = clf.predict(test_data.X)
    return one_hot(self.num_classes, test_pred)

class TFModel(Model):
  def _graph(self):
    role_path = self._resolve_role('train')
    graph = tf.Graph()

    with graph.as_default():
      writer = tf.train.SummaryWriter(role_path)

      train_dataset = tf.placeholder(tf.float32, shape=[None, self.num_features], name='train_dataset')
      train_labels = tf.placeholder(tf.int32, shape=[None, self.num_classes], name='train_labels')
      valid_dataset =  tf.placeholder(tf.float32, shape=[None, self.num_features], name='valid_dataset')
    
      weights_shape = [self.num_features, self.num_classes]

      weights = tf.Variable(
        tf.truncated_normal(weights_shape), name='weights')
      biases = tf.Variable(tf.zeros([self.num_classes]), name='biases')

      def predict(role, dataset):
        return tf.matmul(dataset, weights) + biases

      logits = predict('train', train_dataset)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, train_labels))
    
      train_prediction = tf.nn.softmax(logits, name='train_prediction')
      valid_prediction = tf.nn.softmax(predict('valid', valid_dataset), name='valid_prediction')

      saver = tf.train.Saver()

    return (graph, loss, saver, writer)

  def train(self, train_data, valid_data):
    ckpt_path = self._resolve_model_file('model.ckpt', clean=True)
    alpha = 0.05
    graph, loss, saver, writer = self._graph()

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      writer.add_graph(graph)
      optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
      feed_dict = {'train_dataset:0': train_data.X, 'train_labels:0': one_hot(self.num_classes, train_data.y)}
      summary, train_loss, train_pred = session.run([optimizer, loss, 'train_prediction:0'], feed_dict=feed_dict)
      # TODO don't summarize every step. also summarize test performance every so often
      writer.add_summary(summary, 0)

      [valid_pred] = session.run(['valid_prediction:0'], feed_dict={'valid_dataset:0': valid_data.X})

      saver.save(session, ckpt_path)

    return (train_pred, valid_pred)

  def test(self, test_data):
    ckpt_path = self._resolve_model_file('model.ckpt')
    graph = tf.Graph()
    with tf.Session(graph=graph) as session:
      new_saver = tf.train.import_meta_graph(ckpt_path+'.meta')
      new_saver.restore(session, ckpt_path)
      raw_test_pred = graph.get_tensor_by_name('valid_prediction:0')
      [test_pred] = session.run([raw_test_pred], feed_dict={'valid_dataset:0': test_data.X})
      return test_pred

MODELS = {
  'baseline': BaselineModel,
  'tf': TFModel
}

def make_report(role_path, pred, gold):
  metrics = Metrics(10, pred, gold)
  report = metrics.report()
  filename = os.path.join(role_path, 'report.json')
  write_report(report, filename)

def run_train_model(env, name, variant, train_data, valid_data):
  model = MODELS[name](env, name, variant, train_data.X.shape[1], 10)
  train_pred, valid_pred = model.train(train_data, valid_data)
  make_report(env.resolve_role(name, variant, 'train'), train_pred, train_data.y)
  make_report(env.resolve_role(name, variant, 'valid'), valid_pred, valid_data.y)
  return (train_pred, valid_pred)

def run_test_model(env, name, variant, test_data):
  model = MODELS[name](env, name, variant, test_data.X.shape[1], 10)
  test_pred = model.test(test_data)
  make_report(env.resolve_role(name, variant, 'test'), test_pred, test_data.y)
  return test_pred

def run_load_report(env, name, role):
  filename = env.resolve_role_file(name, variant, 'train', 'report.json')
  return read_report(filename)
