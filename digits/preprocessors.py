"""
Image preprocessors.
"""

import numpy as np
from skimage.color import rgb2gray
import skimage.feature

from .common import product
from .images import img_prepare_all, img_map

# These are all of type function[Data, Data]

def flat_gray(data):
  """
  Turn gray and flatten the dataset.
  """
  X = data.X
  X = rgb2gray(X)
  X = X.astype(np.float32)
  X = X.reshape((X.shape[0], product(X.shape[1:])))
  return data._replace(X=X)

def gray(data):
  """
  Turn the dataset gray (from color).
  """
  X = data.X
  X = rgb2gray(X)
  X = X.astype(np.float32)
  X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
  return data._replace(X=X)

def gray2d(data):
  """
  Turn gray but omit the dummy last dim (HACK).
  """
  X = data.X
  X = rgb2gray(X)
  X = X.astype(np.float32)
  X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
  return data._replace(X=X)

def color(data):
  """
  Do cropping and contrast corrections to the data.
  """
  X = img_prepare_all(data.X)
  return data._replace(X=X)

def hog(data):
  """
  Run the data through HOG.
  """
  X = data.X
  fn = lambda img: skimage.feature.hog(rgb2gray(img), transform_sqrt=True)
  X = img_map(fn, X)
  return data._replace(X=X)

PREPROCESSORS = {
    'noop': lambda data: data,
    'flat-gray': flat_gray,
    'gray': gray,
    'gray-2d': gray2d,
    'color': color,
    'hog': hog
}
