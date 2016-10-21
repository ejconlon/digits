import os
import shutil

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf

from .common import one_hot, un_hot

def run_baseline(name, env, train_data, test_data):
  model = LogisticRegression()
  model.fit(train_data.X, train_data.y)

  train_predict = model.predict(train_data.X)
  train_acc = accuracy_score(train_data.y, train_predict)

  test_predict = model.predict(test_data.X)
  test_acc = accuracy_score(test_data.y, test_predict)

  return (train_acc, test_acc)

def run_tf(name, env, train_data, test_data):
  assert '.' not in name
  logs_path = env.logs
  name_path = os.path.join(logs_path, name)
  if not os.path.isdir(name_path):
    os.makedirs(name_path)
  model_path = os.path.join(name_path, 'model.ckpt')
  if os.path.isfile(model_path):
    os.path.rm(model_path)
  train_path = os.path.join(name_path, 'train')
  if os.path.isdir(train_path):
    shutil.rmtree(train_path)
  test_path = os.path.join(name_path, 'test')
  if os.path.isdir(test_path):
    shutil.rmtree(test_path)

  graph = tf.Graph()
  num_classes = 10
  alpha = 0.5
  num_steps = 100

  with graph.as_default():
    tf_train_dataset = tf.constant(train_data.X)
    tf_train_labels = tf.constant(one_hot(num_classes, train_data.y))
    tf_test_dataset = tf.constant(test_data.X)
  
    weights_shape = [train_data.X.shape[1], num_classes]

    weights = tf.Variable(
      tf.truncated_normal(weights_shape))
    biases = tf.Variable(tf.zeros([num_classes]))

    def predict(role, dataset):
      return tf.matmul(dataset, weights) + biases

    logits = predict('train', tf_train_dataset)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
  
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(predict('test', tf_test_dataset))

    train_writer = tf.train.SummaryWriter(train_path)
    test_writer = tf.train.SummaryWriter(test_path)
    saver = tf.train.Saver()

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()

    train_writer.add_graph(graph)

    for step in range(num_steps):
      summary, train_loss, train_pred0 = session.run([optimizer, loss, train_prediction])
      # TODO don't summarize every step. also summarize test performance every so often
      train_writer.add_summary(summary, step)

    saver.save(session, model_path)

    train_pred = un_hot(num_classes, train_pred0)
    train_acc = accuracy_score(train_pred, train_data.y)
    test_pred = un_hot(num_classes, test_prediction.eval())
    test_acc = accuracy_score(test_pred, test_data.y)

  return (train_acc, test_acc)
