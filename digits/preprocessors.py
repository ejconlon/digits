import numpy as np
from skimage.color import rgb2gray
import skimage.feature

from .common import product
from .images import img_prepare_all

def flat_gray(data):
  X = data.X
  X = rgb2gray(X)
  X = X.astype(np.float32)
  X = X.reshape((X.shape[0], product(X.shape[1:])))
  return data._replace(X=X)

def gray(data):
  X = data.X
  X = rgb2gray(X)
  X = X.astype(np.float32)
  X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
  return data._replace(X=X)

def color(data):
  X = img_prepare_all(data.X)
  return data._replace(X=X)

def hog(data):
  X = data.X
  X = rgb2gray(X)
  X = X.astype(np.float32)
  X = skimage.feature.hog(X, transform_sqrt=True)
  return data._replace(X=X)

PREPROCESSORS = {
    'noop': lambda data: data,
    'flat-gray': flat_gray,
    'gray': gray,
    'color': color,
    'hog': hog
}
