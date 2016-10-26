from abc import ABCMeta, abstractmethod
from collections import namedtuple
import os
import pickle
import shutil

import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

from .common import one_hot, product
from .data import flat_gray, gray

class Model(metaclass=ABCMeta):
  def __init__(self, env, name, variant, num_classes):
    self.env = env
    self.name = name
    self.variant = variant
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
  def preprocess(self, data):
    """ Preprocesses and flattens input data """
    pass

  @abstractmethod
  def train(self, train_data, valid_data):
    """ Return (one_hot preds, one_hot preds) """
    pass

  @abstractmethod
  def test(self, test_data):
    """ Return (one_hot preds) """
    pass

class BaselineModel(Model):
  def preprocess(self, data):
    return flat_gray(data)

  def train(self, train_data, valid_data):
    clf_file = self._resolve_model_file('model.clf', clean=True)
    clf = LogisticRegression()
    train_data = self.preprocess(train_data)
    clf.fit(train_data.X, train_data.y)
    with open(clf_file, 'wb') as f:
      pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    train_pred = clf.predict(train_data.X)
    if valid_data is not None:
      valid_data = self.preprocess(valid_data)
      valid_pred = clf.predict(valid_data.X)
    return (one_hot(self.num_classes, train_pred), one_hot(self.num_classes, valid_pred))

  def test(self, test_data):
    clf_file = self._resolve_model_file('model.clf')
    with open(clf_file, 'rb') as f:
      clf = pickle.load(f)
    test_data = flat_gray(test_data)
    test_pred = clf.predict(test_data.X)
    return one_hot(self.num_classes, test_pred)


# Network definition from
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb
def conv2d(x, W, b, strides=1):
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
  x = tf.nn.bias_add(x, b)
  return tf.nn.relu(x)

def maxpool2d(x, k=2):
  return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(dataset, weights, biases, dropout):
  # Convolution Layer
  conv1 = conv2d(dataset, weights['wc1'], biases['bc1'])
  # Max Pooling (down-sampling)
  conv1 = maxpool2d(conv1, k=2)

  # Convolution Layer
  conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
  # Max Pooling (down-sampling)
  conv2 = maxpool2d(conv2, k=2)

  # Fully connected layer
  # Reshape conv2 output to fit fully connected layer input
  fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
  fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
  fc1 = tf.nn.relu(fc1)
  # Apply Dropout
  fc1 = tf.nn.dropout(fc1, dropout)

  # Output, class prediction
  out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
  return out

# TODO these dimensions probably need massaging :(
def cnn(dataset, dropout, img_width, img_depth, num_classes):
  conv_layers = 2
  feat0 = 16
  feat = lambda n: feat0 * (1 << n)
  c = img_width // (1 << conv_layers)
  unconn = c * c * feat(1)
  conn = 1024

  weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, img_depth, feat(0)])),
    'wc2': tf.Variable(tf.random_normal([5, 5, feat(0), feat(1)])),
    'wd1': tf.Variable(tf.random_normal([unconn, conn])),
    'out': tf.Variable(tf.random_normal([conn, num_classes]))
  }

  biases = {
    'bc1': tf.Variable(tf.random_normal([feat(0)])),
    'bc2': tf.Variable(tf.random_normal([feat(1)])),
    'bd1': tf.Variable(tf.random_normal([conn])),
    'out': tf.Variable(tf.random_normal([num_classes]))
  }

  return conv_net(dataset, weights, biases, dropout)

class TFModel(Model):
  # TODO num_features will be an image with with CNN
  def _graph(self, img_width, img_depth):
    role_path = self._resolve_role('train')
    parent_scope = self._model_name_plus()
    graph = tf.Graph()

    with graph.as_default():
      dataset = tf.placeholder(tf.float32, shape=[None, img_width, img_width, img_depth], name='dataset')
      labels = tf.placeholder(tf.int32, shape=[None, self.num_classes], name='labels')
      keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
      logits = cnn(dataset, keep_prob, img_width, img_depth, self.num_classes)

      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, labels))

      prediction = tf.nn.softmax(logits, name='prediction')
      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

      loss_summary = tf.scalar_summary('loss', loss)
      acc_summary = tf.scalar_summary('accuracy', accuracy)

      summaries = tf.merge_all_summaries()

      writer = tf.train.SummaryWriter(role_path)
      saver = tf.train.Saver()

    return (graph, loss, saver, writer, summaries)

  def preprocess(self, data):
    return gray(data)

  def train(self, train_data, valid_data=None):
    ckpt_path = self._resolve_model_file('model.ckpt', clean=True)

    # Params
    alpha = 0.001
    training_iters = 200000
    batch_size = 256
    display_step = 10
    dropout = 0.90 # keep_prob, 1.0 keep all

    train_data = self.preprocess(train_data)
    assert len(train_data.X.shape) == 4
    img_width = train_data.X.shape[1]
    assert train_data.X.shape[2] == img_width
    img_depth = train_data.X.shape[3]
    graph, loss, saver, writer, summaries = self._graph(img_width, img_depth)
    train_labels = one_hot(self.num_classes, train_data.y)

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      writer.add_graph(graph)
      optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)

      step = 0
      offset = 0
      num_examples = train_data.X.shape[0]

      while step * batch_size < training_iters:
        dataset = train_data.X[offset:offset+batch_size]
        labels = train_labels[offset:offset+batch_size]
        feed_dict = {'dataset:0': dataset, 'labels:0': labels, 'keep_prob:0': dropout}
        session.run([optimizer], feed_dict=feed_dict)
        if step % display_step == 0:
          feed_dict = {'dataset:0': dataset, 'labels:0': labels, 'keep_prob:0': 1.0}
          display_summaries, display_loss, display_acc = session.run([summaries, loss, 'accuracy:0'], feed_dict=feed_dict)
          print('step {} loss {} acc {}'.format(step*batch_size, display_loss, display_acc))
          writer.add_summary(display_summaries, step)
        offset += batch_size
        offset %= num_examples
        step += 1

      print('saving')
      saver.save(session, ckpt_path)

      def batch_pred(dataset, labels):
        preds = []
        offset = 0
        while offset < len(dataset):
          dataset_batch = dataset[offset:offset+batch_size]
          labels_batch = labels[offset:offset+batch_size]
          [pred] = session.run(['prediction:0'], feed_dict={'dataset:0': dataset_batch, 'labels:0': labels_batch, 'keep_prob:0': 1.0})
          preds.append(pred)
          offset += batch_size
        return np.concatenate(preds)

      print('predicting train')
      train_pred = batch_pred(train_data.X, train_labels)
      
      if valid_data is not None:
        print('predicting valid')
        valid_data = self.preprocess(valid_data)
        valid_labels = one_hot(self.num_classes, valid_data.y)
        valid_pred = batch_pred(valid_data.X, valid_labels)
      else:
        valid_pred = None

    return (train_pred, valid_pred)

  def test(self, test_data):
    batch_size = 128
    ckpt_path = self._resolve_model_file('model.ckpt')
    graph = tf.Graph()
    with tf.Session(graph=graph) as session:
      new_saver = tf.train.import_meta_graph(ckpt_path+'.meta')
      new_saver.restore(session, ckpt_path)
      raw_test_pred = graph.get_tensor_by_name('prediction:0')
      test_data = self.preprocess(test_data)
      test_labels = one_hot(self.num_classes, test_data.y)
      def batch_pred(dataset, labels):
        preds = []
        offset = 0
        while offset < len(dataset):
          dataset_batch = dataset[offset:offset+batch_size]
          labels_batch = labels[offset:offset+batch_size]
          [pred] = session.run(['prediction:0'], feed_dict={'dataset:0': dataset_batch, 'labels:0': labels_batch, 'keep_prob:0': 1.0})
          preds.append(pred)
          offset += batch_size
        return np.concatenate(preds)
      return batch_pred(test_data.X, test_labels)

MODELS = {
  'baseline': BaselineModel,
  'tf': TFModel
}

# TODO take num_classes in both of these
def run_train_model(env, name, variant, train_data, valid_data):
  model = MODELS[name](env, name, variant, 10)
  train_pred, valid_pred = model.train(train_data, valid_data)
  return (train_pred, valid_pred)

def run_test_model(env, name, variant, test_data):
  model = MODELS[name](env, name, variant, 10)
  test_pred = model.test(test_data)
  return test_pred
