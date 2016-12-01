"""
Base, SVC, CNN, and Voting classifiers along with the methods to run them.
"""

from abc import ABCMeta, abstractmethod
from argparse import Namespace
from collections import namedtuple
import csv
import gc
import os
import pickle
import random
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.svm import SVC
import tensorflow as tf

from .common import one_hot, product
from .data import invert
from .images import img_select, img_rando, img_width, img_height, img_depth
from .params import PARAMS, SEARCH
from .metrics import Metrics

class Model:
  """
  A base model class to unify sklearn- and tensorflow-based classifiers.

  name and variant are used to resolve paths to artifacts on the filesystem.

  Attributes:
    env (Env): the local filesystem environment
    name (str): the type of model (baseline, tf, vote)
    variant (str): a unique identifier for a run of a classifier
  """
  __metaclass__ = ABCMeta

  def __init__(self, env, name, variant):
    self.env = env
    self.name = name
    self.variant = variant

  def _model_name_plus(self):
    """ Returns: (str) identifier like 'name_variant' """
    return self.env.model_name_plus(self.name, self.variant)

  def _resolve_model(self, clean=False):
    """
    Args:
      clean (bool): remove existing dir
    Returns:
      (str) path to dir for model artifacts
    """
    return self.env.resolve_model(self.name, self.variant, clean)

  def _resolve_role(self, role, clean=False):
    """
    Args:
      clean (bool): remove existing dir
    Returns:
      (str) path to dir for role-specific model artifacts
    """
    return self.env.resolve_role(self.name, self.variant, role, clean)

  def _resolve_model_file(self, filename, clean=False):
    """
    Args:
      clean (bool): remove existing file
    Returns:
      (str) path to given model file
    """
    return self.env.resolve_model_file(self.name, self.variant, filename, clean)

  def _resolve_role_file(self, role, filename, clean=False):
    """
    Args:
      clean (bool): remove existing file
    Returns:
      (str) path to given role-specific model file
    """
    return self.env.resolve_role_file(self.name, self.variant, role, filename, clean)

  @abstractmethod
  def train(self, params, train_data, valid_data=None, max_acc=None):
    """
    Trains this classifier and serialize it and some results to disk.

    Args:
      params (Namespace): bag of model parameters
      train_data (Data): data to train on
      valid_data (Data, optional): data to validate with
      max_acc (float, optional): if present, break after hitting this accuracy
        on the validation data

    Returns:
      (Metrics) metrics on the validation set (including predictions) if valid_data present
    """
    pass

  @abstractmethod
  def test(self, params, test_data):
    """
    Deserializes a trained classifier, tests it, and writes results to disk.

    Args:
      params (Namespace): bag of model parameters
        (HACK: we just need to pull `num_classes` out of this...)
      test_data (Data): data to test

    Returns:
      (Metrics) metrics on the test set (including predictions)
    """
    pass

class BaselineModel(Model):
  """
  sklearn's SVC as a baseline model.
  """

  def train(self, params, train_data, valid_data=None, max_acc=None):
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

def cnn(dataset, dropout, params, width, height, depth):
  """
  Helper function that assembles the graph parts of the CNN.

  Args:
    dataset (tensor): data fed into the network, shaped (N, width, height, depth)
    dropout (float): probability to keep any given connection into FC layers
    params (Namespace): bag of parameters
    width (int): width of individual image in dataset tensor
    height (int): height of individual image in dataset tensor
    depth (int): depth of individual image in dataset tensor

  Returns:
    (tuple) of
      out (tensor): output tensor
      conv_weights (list[tensor]): conv weights by depth
      fc_weights (list[tensor]): fc weights by depth
      out_w (tensor): out weights by depth
      acts (list[tensor]): conv activations by depth
  """
  num_conv = len(params.convs)
  num_fc = len(params.fcs)
  assert num_conv > 0
  assert num_fc > 0

  # TODO use lrn layers? ok for now
  use_lrn = True
  # TODO pool on the last conv layer? ok for now
  pool_last = True

  if pool_last:
    factor = num_conv
  else:
    factor = num_conv - 1

  # calculate conv/fv size
  # width must be evenly divisible by 2**num_conv (or num_conv - 1)
  # because we do 2-pooling after every round
  cw = width // (1 << factor)
  assert cw * (1 << factor) == width
  ch = height // (1 << factor)
  assert ch * (1 << factor) == height
  # unconn is the flattened conv output width going into the FC layer
  unconn = cw * ch * params.convs[-1][1]

  acts = []
  conv_weights = []
  fc_weights = []

  # Xavier seems much worse than just random normal
  # initer = lambda: tf.contrib.layers.xavier_initializer()
  initer = lambda: tf.truncated_normal_initializer()

  conv = dataset
  last_depth = depth
  i = 0
  for (conv_width, conv_depth) in params.convs:
    # Make each conv layer
    w = tf.get_variable(
      "CONV" + str(i),
      shape=[conv_width, conv_width, last_depth, conv_depth],
      initializer=initer()
    )
    b = tf.Variable(tf.random_normal([conv_depth]))
    conv = tf.nn.conv2d(conv, w, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.relu(tf.nn.bias_add(conv, b))
    acts.append(conv)
    if pool_last or (i < num_conv - 1): # optionally skip pool on last layer
      if use_lrn:
        conv = tf.nn.local_response_normalization(conv)
      conv = tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    last_depth = conv_depth
    conv_weights.append(w)
    i += 1

  # Flatten conv output for FC layers
  shape = conv.get_shape().as_list()
  last_conn = shape[1] * shape[2] * shape[3]
  assert last_conn == unconn
  fc = tf.reshape(conv, [-1, last_conn])
  # TODO use dropout between layers? ok for now
  fc = tf.nn.dropout(fc, dropout)

  i = 0
  for conn in params.fcs:
    # Make each FC layer
    w = tf.get_variable(
      "FC" + str(i),
      shape=[last_conn, conn],
      initializer=initer()
    )
    b = tf.Variable(tf.random_normal([conn]))
    fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc, w), b))
    fc = tf.nn.dropout(fc, dropout)
    last_conn = conn
    fc_weights.append(w)
    i += 1

  # our final output layer
  out_w = tf.get_variable(
      "OUT",
      shape=[last_conn, params.num_classes],
      initializer=initer()
    )
  out_b = tf.Variable(tf.random_normal([params.num_classes]))
  out = tf.add(tf.matmul(fc, out_w), out_b)

  return (out, conv_weights, fc_weights, out_w, acts)

def parse_weights(conv_weights):
  """
  Helper function to turn conv weights into a DataFrame for analysis.

  Args:
    conv_weights (list[ndarray]): conv weights by depth

  Returns:
    (DataFrame) table of weights by layer and channel
  """
  w_recs = []
  d = 0
  for ws in conv_weights:
    xs = np.moveaxis(ws, [2,3], [0,1])
    xs = xs.reshape((xs.shape[0]*xs.shape[1], xs.shape[2], xs.shape[3]))
    for i in range(xs.shape[0]):
      w_recs.append({'layer': d, 'channel': i, 'width': xs.shape[2], 'weights': xs[i]})
    d += 1
  return pd.DataFrame.from_records(w_recs, columns=['layer', 'channel', 'width', 'weights'])

def parse_activations(activations):
  """
  Helper function to clean up activation images for analysis.

  Args:
    activations (list[ndarray]): activation images by depth

  Returns:
    (list[ndarray]) same list by depth but with indices in each fixed up
  """
  acts = []
  for orig in activations:
    xs = np.array([orig[i] for i in range(len(orig))])
    xs = np.moveaxis(xs, [3], [1])
    acts.append(xs)
  return acts

class TFModel(Model):
  """
  A tensorflow CNN as a Model.
  """
  def _graph(self, params, width, height, depth):
    """
    Helper to assemble the graph.

    Args:
      params (Namespace): bag of parameters
      width (int): width of each individual image
      height (int): height of each individual image
      depth (int): depth of each individual image

    Returns:
      (tuple) of (graph, loss, saver, writer, summaries, optimizer)
    """
    role_path = self._resolve_role('train')
    parent_scope = self._model_name_plus()
    graph = tf.Graph()

    with graph.as_default():
      dataset = tf.placeholder(tf.float32, shape=[None, width, height, depth], name='dataset')
      labels = tf.placeholder(tf.int32, shape=[None, params.num_classes], name='labels')
      keep_prob = tf.placeholder(tf.float32, name='keep_prob')
      alpha = tf.placeholder(tf.float32, name='alpha')
      decay_step = tf.placeholder(tf.int32, name='decay_step')
      global_step = tf.placeholder(tf.int32, name='global_step')
      decay_factor = tf.placeholder(tf.float32, name='decay_factor')
    
      # Assemble the important parts of the graph (CNN + FC layers into logits)...
      logits, conv_weights, fc_weights, out_w, acts = cnn(dataset, keep_prob, params, width, height, depth)
      # And keep these for analysis!
      for cw in conv_weights:
        tf.add_to_collection('conv_weights', cw)
      for act in acts:
        tf.add_to_collection('activations', act)

      # Regularize!
      reg = sum(tf.nn.l2_loss(w) for w in fc_weights) + tf.nn.l2_loss(out_w)

      # TODO regularize conv weights? ok for now
      reg += sum(tf.nn.l2_loss(w) for w in conv_weights)
      
      # Our loss function is good old cross entropy with a regularization term
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + params.lam * reg

      # We decay learning rate exponentially
      learning_rate = tf.train.exponential_decay(alpha, global_step, decay_step, decay_factor, name='learning_rate')
      # Adam does much better than GD.
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

      # Expose some prediction/accuracy nodes for occasional logging and validation
      prediction = tf.nn.softmax(logits, name='prediction')
      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

      # Summaries to write for tensorboard
      loss_summary = tf.scalar_summary('loss', loss)
      acc_summary = tf.scalar_summary('accuracy', accuracy)

      summaries = tf.merge_all_summaries()

      # writer for tensorboard, saver to de/ser the whole thing
      writer = tf.train.SummaryWriter(role_path)
      saver = tf.train.Saver()

    return (graph, loss, saver, writer, summaries, optimizer)

  def train(self, params, train_data, valid_data=None, max_acc=None):
    ckpt_path = self._resolve_model_file('model.ckpt', clean=True)
    csv_path = self._resolve_model_file('learning_curve.csv')
    # set up logging for a learning curve CSV file
    with open(csv_path, 'w') as csv_file:
      columns = ['step', 'seen', 'train_acc', 'train_loss']
      if valid_data is not None:
        columns.extend(['valid_acc', 'valid_loss'])
      csv_writer = csv.DictWriter(csv_file, columns)
      csv_writer.writeheader()

      # set up data augmentation
      if params.use_rando:
        rando = lambda img: img_rando(img, s=params.rando_scale, r=params.rando_rotation, t=params.rando_translation, i=params.rando_inversion)
      else:
        rando = None
      
      # Figure out how big each image is
      width = img_width(train_data.X)
      height = img_height(train_data.X)
      depth = img_depth(train_data.X)

      # Make the graph
      graph, loss, saver, writer, summaries, optimizer = self._graph(params, width, height, depth)

      # Prepare data
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
        step_offset = random.randint(0, 10000)
        num_examples = train_data.X.shape[0]

        assert params.alpha is not None
        assert params.decay_step is not None
        assert params.decay_factor is not None

        break_acc = 0.0
        break_count = 0
        assert params.break_display_step is None or params.break_display_step > 0

        start_time = time.time()
        try:
          while step * params.batch_size < params.training_iters:   
            #gc.collect()
            print('step', step)
            # Every so often (display_step interval) evaluate, train and test accuracies
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
              # Should we terminate early? Check validation accuracy against parameters
              if valid_data is not None and params.break_display_step is not None:
                acc = row['valid_acc']
                if acc > break_acc:
                  print('accuracy improved to', acc)
                  break_acc = acc
                  break_count = 0
                else:
                  break_count += 1
                  print('not better than', break_acc, '| steps left', params.break_display_step - break_count)
                if break_count >= params.break_display_step:
                  print('breaking early because not improving')
                  break
                if max_acc is not None and acc >= max_acc:
                  print('breaking early because of artificial accuracy limit')
                  break
            # Now train for the round:
            # Sample the training set accordingly
            dataset, labels, _ = img_select(train_data.X, train_labels, train_inv, params.batch_size, rando, params.invert, step + step_offset)
            assert dataset.shape[0] >= params.batch_size
            assert labels.shape[0] == dataset.shape[0]
            feed_dict = {
              'dataset:0': dataset,
              'labels:0': labels,
              'keep_prob:0': params.dropout,
              'alpha:0': params.alpha,
              'decay_factor:0': params.decay_factor,
              'decay_step:0': params.decay_step,
              'global_step:0': step
            }
            # Optimize!
            _, lr = session.run([optimizer, 'learning_rate:0'], feed_dict=feed_dict)
            if step % params.display_step == 0:
              print('learning rate', lr)
            step += 1
        except KeyboardInterrupt:
          # It's handy to be able to Ctrl-C out of training to skip ahead
          print('Caught interrupt. Halting training.')
        end_time = time.time()
        diff_time = end_time - start_time
        print('trained in', diff_time, 'seconds')
        total_seen = step*params.batch_size
        print('saw', total_seen, 'examples of', train_data.X.shape[0], 'total')

        # Write model and weights to disk:

        print('saving')
        saver.save(session, ckpt_path)

        print('writing weights')
        [conv_weights] = session.run([tf.get_collection('conv_weights')])
        conv_weights = parse_weights(conv_weights)
        cw_file = self._resolve_model_file('conv_weights.pickle', clean=True)
        with open(cw_file, 'wb') as f:
          pickle.dump(conv_weights, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Return prediction on validation set:

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
      # Deserialize trained graph
      new_saver = tf.train.import_meta_graph(ckpt_path+'.meta')
      new_saver.restore(session, ckpt_path)
      raw_test_pred = graph.get_tensor_by_name('prediction:0')
      # Prepare data
      test_labels = one_hot(params.num_classes, test_data.y)
      # Return prediction on test set
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

  def activations(self, X):
    """
    Generate activation images for the given dataset.

    Args:
      X (ndarray) array of images

    Returns:
      (list[ndarray]) activation images by depth
    """
    ckpt_path = self._resolve_model_file('model.ckpt')
    graph = tf.Graph()
    with tf.Session(graph=graph) as session:
      new_saver = tf.train.import_meta_graph(ckpt_path+'.meta')
      new_saver.restore(session, ckpt_path)
      feed_dict = {'dataset:0': X}
      [activations] = session.run([tf.get_collection('activations')], feed_dict=feed_dict)
      return parse_activations(activations)

class VoteModel(Model):
  """
  A soft-voting ensemble of CNNs.
  """
  def train(self, params, train_data, valid_data=None, max_acc=None):
    num_models = params.num_classes
    assert num_models > 0
    preds = []
    for i in range(num_models):  
      print('Training sub-model', i)
      model = TFModel(self.env, 'tf', self.variant + '__vote__' + str(i))
      pred = model.train(params, train_data, valid_data, max_acc)
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
      model = TFModel(self.env, 'tf', self.variant + '__vote__' + str(i))
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

def run_train_model(env, name, variant, train_data, valid_data, param_set,
                    search_set=None, search_index=None, max_acc=None):
  """
  Selects parameters and trains a model.

  Args:
    env (Env): the local filesystem environment
    name (str): the type of model (baseline, tf, vote)
    variant (str): a unique identifier for a run of a classifier
    train_data (Data): training data
    valid_data (Data, optional): validation data
    param_set (str): param set identifier (see digits.params.PARAMS)
    search_set (str, optional): search set identifier (see digits.params.SEARCH)
    search_index (int, optional): how many iterations of search we've done
    max_acc (float, optional): if present, break after hitting this accuracy
      on the validation data

  Returns:
    (tuple) of
      params (Namespace): bag of params assembled from param_set and search_set
      valid_metrics (Metrics): metrics on the validation set (including predictions) if valid_data present
  """
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
  valid_pred = model.train(params, train_data, valid_data, max_acc=max_acc)
  if valid_pred is not None:
    valid_metrics = Metrics(params.num_classes, valid_pred, valid_data.y)
  else:
    valid_metrics = None
  return (params, valid_metrics)

def run_test_model(env, name, variant, test_data, param_set):
  """
  Tests a model.

  Args:
    env (Env): the local filesystem environment
    name (str): the type of model (baseline, tf, vote)
    variant (str): a unique identifier for a run of a classifier
    test_data (Data): test data
    param_set (str): param set identifier (see digits.params.PARAMS)

  Returns:
    (Metrics) metrics on the validation set (including predictions)
  """
  model = MODELS[name](env, name, variant)
  params = PARAMS[name][param_set]
  test_pred = model.test(params, test_data)
  test_metrics = Metrics(params.num_classes, test_pred, test_data.y)
  return test_metrics
