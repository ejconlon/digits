import os
import pickle
import shutil

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf

from .common import one_hot

def prepare(env, name, roles):
  assert '.' not in name
  logs_path = env.logs
  name_path = os.path.join(logs_path, name)
  if not os.path.isdir(name_path):
    os.makedirs(name_path)
  # per-role output dirs
  paths = dict((role, os.path.join(name_path, role)) for role in roles)
  for path in paths.values():
    if os.path.isdir(path):
      shutil.rmtree(path)
  # train artifact files
  artifacts = ['ckpt', 'clf']
  artifact_paths = dict((art, os.path.join(name_path, 'model.' + art)) for art in artifacts)
  if 'train' in roles:
    for path in artifact_paths.values():
      if os.path.isfile(path):
        os.remove(path)
  paths.update(artifact_paths)
  return paths

def train_baseline(paths, train_data, valid_data):
  num_classes = 10
  model = LogisticRegression()
  model.fit(train_data.X, train_data.y)
  with open(paths['clf'], 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
  train_pred = model.predict(train_data.X)
  valid_pred = model.predict(valid_data.X)
  return (one_hot(num_classes, train_pred), one_hot(num_classes, valid_pred))

def test_baseline(paths, test_data):
  num_classes = 10
  with open(paths['clf'], 'rb') as f:
    model = pickle.load(f)
  test_pred = model.predict(test_data.X)
  return one_hot(num_classes, test_pred)

def train_tf(paths, train_data, valid_data):
  graph = tf.Graph()
  num_features = train_data.X.shape[1]
  num_classes = 10
  alpha = 0.5
  num_steps = 100

  with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=[None, num_features], name='train_dataset')
    tf_train_labels = tf.placeholder(tf.int32, shape=[None, num_classes], name='train_labels')
    tf_valid_dataset =  tf.placeholder(tf.float32, shape=[None, num_features], name='valid_dataset')
  
    weights_shape = [num_features, num_classes]

    weights = tf.Variable(
      tf.truncated_normal(weights_shape), name='weights')
    biases = tf.Variable(tf.zeros([num_classes]), name='biases')

    def predict(role, dataset):
      return tf.matmul(dataset, weights) + biases

    logits = predict('train', tf_train_dataset)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
  
    train_prediction = tf.nn.softmax(logits, name='train_prediction')
    valid_prediction = tf.nn.softmax(predict('valid', tf_valid_dataset), name='valid_prediction')

    train_writer = tf.train.SummaryWriter(paths['train'])
    valid_writer = tf.train.SummaryWriter(paths['valid'])
    saver = tf.train.Saver()

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    train_writer.add_graph(graph)
    feed_dict = {'train_dataset:0': train_data.X, 'train_labels:0': one_hot(num_classes, train_data.y)}
    summary, train_loss, train_pred = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    # TODO don't summarize every step. also summarize test performance every so often
    train_writer.add_summary(summary, 0)

    [valid_pred] = session.run([valid_prediction], feed_dict={tf_valid_dataset: valid_data.X})

    saver.save(session, paths['ckpt'])

  return (train_pred, valid_pred)

def test_tf(paths, test_data):
  num_classes = 10
  graph = tf.Graph()
  with tf.Session(graph=graph) as session:
    new_saver = tf.train.import_meta_graph(paths['ckpt']+'.meta')
    new_saver.restore(session, paths['ckpt'])
    raw_test_pred = graph.get_tensor_by_name('valid_prediction:0')
    [test_pred] = session.run([raw_test_pred], feed_dict={'valid_dataset:0': test_data.X})
    return test_pred

MODELS = {
  'baseline': (train_baseline, test_baseline),
  'tf': (train_tf, test_tf)
}

def train_model(env, name, train_data, valid_data):
  assert name in MODELS
  paths = prepare(env, name, ['train', 'valid'])
  return MODELS[name][0](paths, train_data, valid_data)

def test_model(env, name, test_data):
  assert name in MODELS
  paths = prepare(env, name, ['test'])
  return MODELS[name][1](paths, test_data)

def train_and_test_model(env, name, train_data, valid_data, test_data):
  assert name in MODELS
  paths = prepare(env, name, ['train', 'valid', 'test'])
  train_pred, valid_pred = MODELS[name][0](paths, train_data, valid_data)
  test_pred = MODELS[name][1](paths, test_data)
  return (train_pred, valid_pred, test_pred)
