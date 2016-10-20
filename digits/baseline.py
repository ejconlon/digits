from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tensorflow as tf
import tensorflow.contrib.learn as tflearn

def run_baseline(train_data, test_data):
  model = LogisticRegression()
  model.fit(train_data.X, train_data.y)
  train_predict = model.predict(train_data.X)
  train_acc = accuracy_score(train_data.y, train_predict)
  test_predict = model.predict(test_data.X)
  test_acc = accuracy_score(test_data.y, test_predict)
  return (model, train_acc, test_acc)

def run_tf(train_data, test_data):
  feature_columns = tflearn.infer_real_valued_columns_from_input(train_data.X)
  model = tflearn.LinearClassifier(n_classes=10, feature_columns=feature_columns)
  return (model, 0, 0)
