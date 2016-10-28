import random
import warnings

import numpy as np
import skimage.filters.rank
import skimage.exposure
import skimage.morphology
import skimage.transform

from .common import product

DEFAULT_S=1.03
DEFAULT_R=0.15
DEFAULT_T=1.5
DEFAULT_I=0.5

def img_width(arr):
  if is_single_img(arr):
    i = 0
  else:
    i = 1
  assert arr.shape[i] == arr.shape[i + 1]
  return arr.shape[i]

def img_depth(arr):
  if len(arr.shape) == 2:
    return 1
  else:
    if len(arr.shape) == 3:
      i = 2
    else:
      assert len(arr.shape) == 4
      i = 3
    d = arr.shape[i]
    if d == 1 or d == 3:
      return d
    else:
      return 1

# is the array a single image or multiple?
def is_single_img(arr):
  return len(arr.shape) == 2 or \
    (len(arr.shape) == 3 and (arr.shape[2] == 1 or arr.shape[2] == 3))

# apply a function intelligently to an image or array of images
def img_map(f, arr):
  if is_single_img(arr):
    return f(arr)
  else:
    return np.stack([f(arr[i]) for i in range(arr.shape[0])])

def img_map_id(f, arr):
  out = np.empty(arr.shape)
  for i in range(arr.shape[0]):
    f(arr[i], out[i])
  return out

# apply a function for side effects
def img_effect(f, arr):
  if is_single_img(arr):
    f(arr)
  else:
    for i in range(arr.shape[0]):
      f(arr[i])

def img_flatten(arr):
  assert len(arr.shape) == 4
  return arr.reshape((arr.shape[0], product(arr.shape[1:])))

def img_flatten(arr):
  width = img_width(arr)
  depth = img_depth(arr)
  volume = width * width * depth
  return (arr.reshape((-1, volume)), width, depth)

def img_unflatten(arr, width, depth):
  if depth == 1:
    return arr.reshape((-1, width, width))
  else:
    assert depth == 3
    return arr.reshape((-1, width, width, depth))

# apply a random transformation to an image
def img_rando(img, s=DEFAULT_S, r=DEFAULT_R, t=DEFAULT_T, i=DEFAULT_I):
  scale = random.uniform(1/s, s)
  rot = random.uniform(-r, r)
  trans = (random.uniform(-t, t), random.uniform(-t, t))
  tform = skimage.transform.SimilarityTransform(scale=scale, rotation=rot, translation=trans)
  warped = skimage.transform.warp(img, tform, mode='symmetric')
  if random.random() < i:
    warped = 1.0 - warped
  return warped

def img_contrast(img, out, selem, p):
  return skimage.filters.rank.enhance_contrast_percentile(img, out=out, selem=selem, p0=p, p1=1.0-p)

def img_contrast_all(arr):
  p = 0.2
  c = 3
  selem = skimage.morphology.square(c)  
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return img_map_id(lambda img, out: img_contrast(img, out, selem, p), arr)

def img_prepare_all(arr):
  if len(arr.shape) == 3:
    arr = skimage.img_as_float(arr)
  else:
    assert len(arr.shape) == 4
    arr = skimage.color.rgb2gray(arr)
  assert len(arr.shape) == 3
  gray_shape = list(arr.shape)
  gray_shape.append(1)
  #arr = img_contrast_all(arr)
  arr = arr.reshape(gray_shape)
  return arr

def img_select(X, y, batch_size, augment=None):
  assert X.shape[0] == y.shape[0]
  lim = X.shape[0]
  Xb_shape = list(X.shape)
  Xb_shape[0] = batch_size
  Xb = np.empty(Xb_shape, dtype=X.dtype)
  yb_shape = list(y.shape)
  yb_shape[0] = batch_size
  yb = np.empty(yb_shape, dtype=y.dtype)
  for i in range(batch_size):
    index = random.randint(0, lim-1)
    if augment is None:
      Xb[i] = X[index]
    else:
      Xb[i] = augment(X[index])
    yb[i] = y[index]
  return (Xb, yb)
