import numpy as np
from skimage.color import rgb2gray

from .common import product
from .images import img_prepare_all

def flat_gray(data):
  X = data.X
  X = rgb2gray(X)
  X = X.astype(np.float32)
  X = X.reshape((X.shape[0], product(X.shape[1:])))
  new_data = data._replace(X=X)
  return new_data

def gray(data):
  X = data.X
  X = rgb2gray(X)
  X = X.astype(np.float32)
  X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
  new_data = data._replace(X=X)
  return new_data

def color(data):
  X = img_prepare_all(data.X)
  return data._replace(X=X)

PREPROCESSORS = {
    'flat-gray': flat_gray,
    'gray': gray,
    'color': color
}
