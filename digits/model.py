from abc import ABCMeta, abstractclass

class Model(ABCMeta):
  @abstractmethod
  def fit(self, X, y, *args, **kwargs):
    pass

  @abstractmethod
  def predict(self, X):
    pass

  @abstractmethod
  def score(self, X, y):
    pass

class TFModel(Model):
  def fit(self, X, y):
    pass

  def predict(self, X, y):
    pass

  def score(self, X, y):
    pass