from abc import ABCMeta, abstractmethod
from collections import namedtuple
import os
import pickle
import shutil

import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

from .common import one_hot

class Model(metaclass=ABCMeta):
  def __init__(self, env, name, variant, num_features, num_classes):
    self.env = env
    self.name = name
    self.variant = variant
    self.num_features = num_features
    self.num_classes = num_classes

  def _model_name_plus(self):
    return self.env.model_name_plus(self.name, self.variant)

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
    parent_scope = self._model_name_plus()
    graph = tf.Graph()

    with graph.as_default():
      dataset = tf.placeholder(tf.float32, shape=[None, self.num_features], name='dataset')
      labels = tf.placeholder(tf.int32, shape=[None, self.num_classes], name='labels')
      keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
      weights_shape = [self.num_features, self.num_classes]

      weights = tf.Variable(
        tf.truncated_normal(weights_shape), name='weights')
      biases = tf.Variable(tf.zeros([self.num_classes]), name='biases')

      def predict(role, dataset):
        return tf.matmul(dataset, weights) + biases

      logits = predict('train', dataset)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, labels))

      prediction = tf.nn.softmax(logits, name='prediction')
      #correct = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(y, 1))
      #accuracy = tf.reduce_mean(tf.cast(train_correct, tf.float32), name='accuracy')

      writer = tf.train.SummaryWriter(role_path)
      saver = tf.train.Saver()

    return (graph, loss, saver, writer)

  def train(self, train_data, valid_data=None):
    ckpt_path = self._resolve_model_file('model.ckpt', clean=True)

    # Params
    alpha = 0.001
    training_iters = 200000
    batch_size = 128
    display_step = 10
    dropout = 0.75 # keep_prob, 1.0 keep all

    graph, loss, saver, writer = self._graph()
    train_labels = one_hot(self.num_classes, train_data.y)

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      loss_summary = tf.scalar_summary('loss', loss)
      writer.add_graph(graph)
      optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

      step = 0
      offset = 0
      num_examples = train_data.X.shape[0]

      while step * batch_size < training_iters:
        dataset = train_data.X[offset:offset+batch_size]
        labels = train_labels[offset:offset+batch_size]
        feed_dict = {'dataset:0': train_data.X, 'labels:0': train_labels, 'keep_prob:0': dropout}
        act_opt_summary, act_loss_summary, train_pred = session.run([optimizer, loss_summary, 'prediction:0'], feed_dict=feed_dict)
        if step % display_step == 0:
          # TODO calculate accuracy too
          print("step", step)
          #writer.add_summary(act_opt_summary, step)
          writer.add_summary(act_loss_summary, step)
        offset += batch_size
        offset %= num_examples
        step += 1

      if valid_data is not None:
        valid_labels = one_hot(self.num_classes, valid_data.y)
        [valid_pred] = session.run(['prediction:0'], feed_dict={'dataset:0': valid_data.X, 'labels:0': valid_labels, 'keep_prob:0': 1.0})
      else:
        valid_pred = None

      saver.save(session, ckpt_path)

    return (train_pred, valid_pred)

  def test(self, test_data):
    ckpt_path = self._resolve_model_file('model.ckpt')
    graph = tf.Graph()
    with tf.Session(graph=graph) as session:
      new_saver = tf.train.import_meta_graph(ckpt_path+'.meta')
      new_saver.restore(session, ckpt_path)
      raw_test_pred = graph.get_tensor_by_name('prediction:0')
      test_labels = one_hot(self.num_classes, test_data.y)
      [test_pred] = session.run([raw_test_pred], feed_dict={'dataset:0': test_data.X, 'labels:0': test_labels, 'keep_prob:0': 1.0})
      return test_pred

MODELS = {
  'baseline': BaselineModel,
  'tf': TFModel
}

def run_train_model(env, name, variant, train_data, valid_data):
  model = MODELS[name](env, name, variant, train_data.X.shape[1], 10)
  train_pred, valid_pred = model.train(train_data, valid_data)
  return (train_pred, valid_pred)

def run_test_model(env, name, variant, test_data):
  model = MODELS[name](env, name, variant, test_data.X.shape[1], 10)
  test_pred = model.test(test_data)
  return test_pred
