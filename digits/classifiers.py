import os
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
  paths = dict((role, os.path.join(name_path, role)) for role in roles)
  for path in paths.values():
    if os.path.isdir(path):
      shutil.rmtree(path)
  ckpt_path = os.path.join(name_path, 'model.ckpt')
  if 'train' in roles and os.path.isfile(ckpt_path):
    os.remove(ckpt_path)
  paths['ckpt'] = ckpt_path
  return paths

def train_baseline(paths, train_data):
  num_classes = 10
  model = LogisticRegression()
  model.fit(train_data.X, train_data.y)
  train_pred = model.predict(train_data.X)
  return (model, one_hot(num_classes, train_pred))

def test_baseline(paths, model, test_data):
  num_classes = 10
  test_pred = model.predict(test_data.X)
  return one_hot(num_classes, test_pred)

def train_tf(paths, train_data):
  graph = tf.Graph()
  num_features = train_data.X.shape[1]
  num_classes = 10
  alpha = 0.5
  num_steps = 100

  with graph.as_default():
    # TODO make these placeholders
    tf_train_dataset = tf.constant(train_data.X, name='train_dataset')
    tf_train_labels = tf.constant(one_hot(num_classes, train_data.y), name='train_labels')
    tf_test_dataset =  tf.placeholder(tf.float32, shape=[None, num_features], name='test_dataset')
  
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
    test_prediction = tf.nn.softmax(predict('test', tf_test_dataset), name='test_prediction')

    train_writer = tf.train.SummaryWriter(paths['train'])
    test_writer = tf.train.SummaryWriter(paths['test'])
    saver = tf.train.Saver()

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    train_writer.add_graph(graph)

    for step in range(num_steps):
      summary, train_loss, train_pred = session.run([optimizer, loss, train_prediction])
      # TODO don't summarize every step. also summarize test performance every so often
      train_writer.add_summary(summary, step)

    saver.save(session, paths['ckpt'])

  return (graph, train_pred)

def test_tf(paths, graph, test_data):
  num_classes = 10
  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    raw_test_pred = graph.get_tensor_by_name('test_prediction:0')
    result = session.run([raw_test_pred], feed_dict={'test_dataset:0': test_data.X})
    return result[0]

MODELS = {
  'baseline': (train_baseline, test_baseline),
  'tf': (train_tf, test_tf)
}

def train_model(env, name, train_data):
  assert name in MODELS
  paths = prepare(env, name, ['train'])
  return MODELS[name][0](paths, train_data)

def test_model(env, name, model, test_data):
  assert name in MODELS
  paths = prepare(env, name, ['test'])
  return MODELS[name][1](paths, model, test_data)

def train_and_test_model(env, name, train_data, test_data):
  assert name in MODELS
  paths = prepare(env, name, ['train', 'test'])
  model, train_pred = MODELS[name][0](paths, train_data)
  test_pred = MODELS[name][1](paths, model, test_data)
  return (model, train_pred, test_pred)
