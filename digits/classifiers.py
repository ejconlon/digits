from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tensorflow as tf

def run_baseline(train_data, test_data):
  model = LogisticRegression()
  model.fit(train_data.X, train_data.y)

  train_predict = model.predict(train_data.X)
  train_acc = accuracy_score(train_data.y, train_predict)

  test_predict = model.predict(test_data.X)
  test_acc = accuracy_score(test_data.y, test_predict)

  return (train_acc, test_acc)

def run_tf(train_data, test_data):
  graph = tf.Graph()
  num_labels = 10
  alpha = 0.5
  num_steps = 100

  with graph.as_default():
    tf_train_dataset = tf.constant(train_data.X)
    tf_train_labels = tf.constant(train_data.y)
    tf_test_dataset = tf.constant(test_data.X)
  
    weights = tf.Variable(
      tf.truncated_normal([train_data.X.shape[1], num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    def predict(role, dataset):
      return tf.matmul(dataset, weights) + biases

    logits = predict('train', tf_train_dataset)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  
    optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
  
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(predict('test', tf_test_dataset))

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    for step in range(num_steps):
      _, l, predictions = session.run([optimizer, loss, train_prediction])
    train_acc = accuracy_score(train_prediction.eval(), train_data.y)
    test_acc = accuracy_score(test_prediction.eval(), test_data.y)

  return (train_acc, test_acc)
