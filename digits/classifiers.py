from abc import ABCMeta, abstractmethod
from argparse import Namespace
from collections import namedtuple
import csv
import gc
import os
import pickle
import random
import shutil

import numpy as np
from sklearn.svm import SVC
import tensorflow as tf

from .common import one_hot, product
from .data import invert
from .images import img_select, img_rando, img_width, img_height, img_depth
from .params import PARAMS, SEARCH
from .metrics import Metrics

class Model:
  __metaclass__ = ABCMeta

  def __init__(self, env, name, variant):
    self.env = env
    self.name = name
    self.variant = variant

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
  def train(self, params, train_data, valid_data=None):
    """ Return (train preds, valid preds) """
    pass

  @abstractmethod
  def test(self, params, test_data):
    """ Return (test preds) """
    pass

class BaselineModel(Model):
  def train(self, params, train_data, valid_data=None):
    clf_file = self._resolve_model_file('model.clf', clean=True)
    clf = SVC()
    print('baseline fitting')
    clf.fit(train_data.X, train_data.y)
    print('baseline pickling')
    with open(clf_file, 'wb') as f:
      pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    if valid_data is not None:
      print('baseline predicting valid')
      valid_pred = clf.predict(valid_data.X)
      return one_hot(params.num_classes, valid_pred)
    else:
      return None

  def test(self, params, test_data):
    clf_file = self._resolve_model_file('model.clf')
    print('baseline unpickling')
    with open(clf_file, 'rb') as f:
      clf = pickle.load(f)
    print('baseline predicting test')
    test_pred = clf.predict(test_data.X)
    return one_hot(params.num_classes, test_pred)


# conv2d/maxpool2d definition from
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network.ipynb
def conv2d(x, W, b, strides=1):
  x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
  x = tf.nn.bias_add(x, b)
  return tf.nn.relu(x)

def maxpool2d(x, k=2):
  return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def cnn(dataset, dropout, params, width, height, depth):
  num_conv = len(params.convs)
  num_fc = len(params.fcs)
  assert num_conv > 0
  assert num_fc > 0

  # calculate conv/fv size
  # width must be evenly divisible by 2**num_conv
  # because we do 2-pooling after every round (except last)
  cw = width // (1 << (num_conv - 1))
  assert cw * (1 << (num_conv - 1)) == width
  ch = height // (1 << (num_conv - 1))
  assert ch * (1 << (num_conv - 1)) == height
  unconn = cw * ch * params.convs[-1][1]

  conv_weights = []
  fc_weights = []

  conv = dataset
  last_depth = depth
  i = 0
  for (conv_width, conv_depth) in params.convs:
    w = tf.Variable(tf.random_normal([conv_width, conv_width, last_depth, conv_depth]))
    b = tf.Variable(tf.random_normal([conv_depth]))
    conv = conv2d(conv, w, b)
    if i != len(params.convs) - 1:
      conv = tf.nn.local_response_normalization(conv)
      conv = maxpool2d(conv, k=2)
    last_depth = conv_depth
    conv_weights.append(w)
    i += 1

  fc = tf.nn.dropout(tf.reshape(conv, [-1, unconn]), dropout)

  last_conn = unconn
  for conn in params.fcs:
    w = tf.Variable(tf.random_normal([last_conn, conn]))
    b = tf.Variable(tf.random_normal([conn]))
    fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc, w), b))
    last_conn = conn
    fc_weights.append(w)

  out_w = tf.Variable(tf.random_normal([last_conn, params.num_classes]))
  out_b = tf.Variable(tf.random_normal([params.num_classes]))

  out = tf.add(tf.matmul(fc, out_w), out_b)

  return (out, conv_weights, fc_weights)

class TFModel(Model):
  def _graph(self, params, width, height, depth):
    role_path = self._resolve_role('train')
    parent_scope = self._model_name_plus()
    graph = tf.Graph()

    with graph.as_default():
      dataset = tf.placeholder(tf.float32, shape=[None, width, height, depth], name='dataset')
      labels = tf.placeholder(tf.int32, shape=[None, params.num_classes], name='labels')
      keep_prob = tf.placeholder(tf.float32, name='keep_prob')
      alpha = tf.placeholder(tf.float32, name='alpha')
      global_step = tf.placeholder(tf.int32, name='global_step')
    
      logits, conv_weights, fc_weights = cnn(dataset, keep_prob, params, width, height, depth)

      reg = sum(tf.nn.l2_loss(w) for w in conv_weights) + \
            sum(tf.nn.l2_loss(w) for w in fc_weights)

      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + params.lam * reg

      optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

      prediction = tf.nn.softmax(logits, name='prediction')
      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

      loss_summary = tf.scalar_summary('loss', loss)
      acc_summary = tf.scalar_summary('accuracy', accuracy)

      summaries = tf.merge_all_summaries()

      writer = tf.train.SummaryWriter(role_path)
      saver = tf.train.Saver()

    return (graph, loss, saver, writer, summaries, optimizer)

  def train(self, params, train_data, valid_data=None):
    ckpt_path = self._resolve_model_file('model.ckpt', clean=True)
    csv_path = self._resolve_model_file('learning_curve.csv')
    with open(csv_path, 'w') as csv_file:
      columns = ['step', 'seen', 'train_acc', 'train_loss']
      if valid_data is not None:
        columns.extend(['valid_acc', 'valid_loss'])
      csv_writer = csv.DictWriter(csv_file, columns)
      csv_writer.writeheader()

      if params.use_rando:
        # TODO tune rando params
        rando = lambda img: img_rando(img, s=params.rando_scale, r=params.rando_rotation, t=params.rando_translation, i=params.rando_inversion)
      else:
        rando = None
      
      width = img_width(train_data.X)
      height = img_height(train_data.X)
      depth = img_depth(train_data.X)
      graph, loss, saver, writer, summaries, optimizer = self._graph(params, width, height, depth)
      train_labels = one_hot(params.num_classes, train_data.y)
      train_inv = invert(params.num_classes, train_data.y)
      if valid_data is not None:
        valid_labels = one_hot(params.num_classes, valid_data.y)
        valid_inv = invert(params.num_classes, valid_data.y)
      else:
        valid_labels = None
        valid_inv = None

      with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        writer.add_graph(graph)

        step = 0
        num_examples = train_data.X.shape[0]

        assert params.alpha is not None
        assert (params.decay_step is None and params.decay_factor is None) or \
               (params.decay_step is not None and params.decay_factor is not None)
        alpha = params.alpha
        print('initial alpha', alpha)

        break_acc = 0.0
        break_count = 0
        assert params.break_display_step is None or params.break_display_step > 0

        try:
          while step * params.batch_size < params.training_iters:   
            gc.collect()
            print('step', step)
            # First evaluate
            if step % params.display_step == 0:
              seen = step * params.batch_size
              row = { 'step': step, 'seen': seen }
              sets = [('train', img_select(train_data.X, train_labels, train_inv, params.display_size))]
              if valid_data is not None:
                sets.append(('valid', img_select(valid_data.X, valid_labels, valid_inv, params.display_size)))
              for (role, (dataset, labels, _)) in sets:
                feed_dict = {'dataset:0': dataset, 'labels:0': labels, 'keep_prob:0': 1.0}
                display_summaries, display_loss, display_acc = session.run([summaries, loss, 'accuracy:0'], feed_dict=feed_dict)
                display_loss, display_acc = session.run([loss, 'accuracy:0'], feed_dict=feed_dict)
                print('batch {} seen {} role {} loss {} acc {}'.format(step, seen, role, display_loss, display_acc))
                row[role + '_loss'] = display_loss
                row[role + '_acc'] = display_acc
              #writer.add_summary(display_summaries, step)
              csv_writer.writerow(row)
              if valid_data is not None and params.break_display_step is not None:
                acc = row['valid_acc']
                if acc > break_acc:
                  print('accuracy improved to', acc)
                  break_acc = acc
                  break_count = 0
                else:
                  break_count += 1
                if break_count >= params.break_display_step:
                  print('breaking early because not improving')
                  break
                if params.max_acc is not None and acc >= params.max_acc:
                  print('breaking early because of artificial accuracy limit')
                  break
            # Now train for the round
            dataset, labels, _ = img_select(train_data.X, train_labels, train_inv, params.batch_size, rando)
            feed_dict = {
              'dataset:0': dataset,
              'labels:0': labels,
              'keep_prob:0': params.dropout,
              'alpha:0': alpha,
              'global_step:0': step
            }
            session.run([optimizer], feed_dict=feed_dict)
            if params.decay_step is not None:
              if step > 0 and step % params.decay_step == 0:
                alpha *= params.decay_factor
                print('decaying alpha to', alpha)
            step += 1
        except KeyboardInterrupt:
          print('Caught interrupt. Halting training.')

        print('saving')
        saver.save(session, ckpt_path)

        def batch_pred(dataset, labels):
          preds = []
          offset = 0
          while offset < len(dataset):
            dataset_batch = dataset[offset:offset+params.display_size]
            labels_batch = labels[offset:offset+params.display_size]
            [pred] = session.run(['prediction:0'], feed_dict={'dataset:0': dataset_batch, 'labels:0': labels_batch, 'keep_prob:0': 1.0})
            preds.append(pred)
            offset += params.display_size
          return np.concatenate(preds)

        if valid_data is not None:
          print('predicting valid')
          valid_pred = batch_pred(valid_data.X, valid_labels)
        else:
          valid_pred = None

      return valid_pred

  def test(self, params, test_data):
    ckpt_path = self._resolve_model_file('model.ckpt')
    graph = tf.Graph()
    with tf.Session(graph=graph) as session:
      new_saver = tf.train.import_meta_graph(ckpt_path+'.meta')
      new_saver.restore(session, ckpt_path)
      raw_test_pred = graph.get_tensor_by_name('prediction:0')
      test_labels = one_hot(params.num_classes, test_data.y)
      def batch_pred(dataset, labels):
        preds = []
        offset = 0
        while offset < len(dataset):
          dataset_batch = dataset[offset:offset+params.display_size]
          labels_batch = labels[offset:offset+params.display_size]
          [pred] = session.run(['prediction:0'], feed_dict={'dataset:0': dataset_batch, 'labels:0': labels_batch, 'keep_prob:0': 1.0})
          preds.append(pred)
          offset += params.display_size
        return np.concatenate(preds)
      return batch_pred(test_data.X, test_labels)

class VoteModel(Model):
  def train(self, params, train_data, valid_data=None):
    num_models = params.num_classes
    assert num_models > 0
    preds = []
    for i in range(num_models):  
      print('Training sub-model', i)
      model = TFModel(self.env, 'vote', self.variant + '__' + str(i))
      pred = model.train(params, train_data, valid_data)
      if valid_data is not None:
        assert pred is not None
        preds.append(pred)
    if valid_data is not None:
      assert len(preds) == num_models
      return np.mean(np.array(preds), axis=0)
    else:
      return None

  def test(self, params, test_data):
    num_models = params.num_classes
    assert num_models > 0
    preds = []
    for i in range(num_models): 
      print('Testing sub-model', i) 
      model = TFModel(self.env, 'vote', self.variant + '__' + str(i))
      pred = model.test(params, test_data)
      assert pred is not None
      preds.append(pred)
    assert len(preds) == num_models
    return np.mean(np.array(preds), axis=0)

MODELS = {
  'baseline': BaselineModel,
  'tf': TFModel,
  'vote': VoteModel
}

def run_train_model(env, name, variant, train_data, valid_data, param_set, search_set=None, search_index=None):
  model = MODELS[name](env, name, variant)
  orig_params = PARAMS[name][param_set]  
  if search_set is not None:
    search = SEARCH[name][search_set]
    params = Namespace()
    for (k, v) in orig_params._get_kwargs():
      setattr(params, k, v)
    if type(search) == list:
      override_params = search[search_index]
      for (k, v) in override_params._get_kwargs():
        setattr(params, k, v)
    else:
      for (k, vs) in search._get_kwargs():
        setattr(params, k, random.choice(vs))
  else:
    params = orig_params
  valid_pred = model.train(params, train_data, valid_data)
  if valid_pred is not None:
    valid_metrics = Metrics(params.num_classes, valid_pred, valid_data.y)
  else:
    valid_metrics = None
  return (params, valid_metrics)

def run_test_model(env, name, variant, test_data, param_set):
  model = MODELS[name](env, name, variant)
  params = PARAMS[name][param_set]
  test_pred = model.test(params, test_data)
  test_metrics = Metrics(params.num_classes, test_pred, test_data.y)
  return test_metrics
