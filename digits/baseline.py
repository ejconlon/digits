from sklearn.linear_model import LogisticRegression

def run_baseline(train_data, test_data):
  model = LogisticRegression()
  model.fit(train_data.X, train_data.y)
  train_score = model.score(train_data.X, train_data.y)
  test_score = model.score(test_data.X, test_data.y)
  return (model, train_score, test_score)
