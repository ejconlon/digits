import random
import warnings

import numpy as np
import skimage.color
import skimage.filters.rank
import skimage.exposure
import skimage.morphology
import skimage.transform

from .common import product

# Scale limit (1.0 is identity)
DEFAULT_SCALE=1.02
# Rotation limit (0 is identity)
DEFAULT_ROTATION=0.10
# Translation limit (0 is identity)
DEFAULT_TRANSLATION=1.1
# Inversion prob (0 is identity)
DEFAULT_INVERSION=0.0

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
    v0 = f(arr[0])
    s = list(v0.shape)
    s.insert(0, arr.shape[0])
    v = np.empty(s)
    v[0] = v0
    for i in range(1, arr.shape[0]):
      if i > 0 and i % 10000 == 0:
        print('processing', i)
      vi = f(arr[i])
      v[i] = vi
    return v

def img_map_id(f, arr):
  out = np.empty(arr.shape)
  for i in range(arr.shape[0]):
    if i > 0 and i % 10000 == 0:
      print('processing', i)
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
def img_rando(img, s=DEFAULT_SCALE, r=DEFAULT_ROTATION, t=DEFAULT_TRANSLATION, i=DEFAULT_INVERSION):
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

def img_gray_contrast_all(arr):
  p = 0.2
  k = 3
  selem = skimage.morphology.square(k)  
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    return img_map_id(lambda img, out: img_contrast(img, out, selem, p), arr)

def img_color_contrast_all(arr):
  k = 7
  c = 0.01
  # gray avg
  selem = skimage.morphology.square(k)
  def fn(img):
    img = skimage.color.rgb2gray(img)
    img = skimage.filters.rank.subtract_mean(img, selem)
    img = skimage.img_as_float(img)
    img = skimage.exposure.rescale_intensity(img)
    return img
  # only hsv
  #fn = lambda img: skimage.color.rgb2hsv(img)
  # only gray
  #fn = lambda img: skimage.color.rgb2gray(img)
  return img_map(fn, arr)

def img_prepare_all(arr):
  if len(arr.shape) == 3:
    # gray (mnist)
    arr = skimage.img_as_float(arr)
  else:
    # color (svhn)
    assert len(arr.shape) == 4
    arr = img_color_contrast_all(arr)  # not necessary for mnist
  if len(arr.shape) == 3:
    gray_shape = list(arr.shape)
    gray_shape.append(1)
    arr = arr.reshape(gray_shape)
  assert len(arr.shape) == 4
  return arr

def img_select(X, y, y_inv, batch_size, augment=None):
  assert X.shape[0] == y.shape[0]
  num_classes = len(y_inv)
  per_class = batch_size // num_classes
  assert num_classes * per_class == batch_size
  seen = [0 for i in range(num_classes)]
  avail = list(range(num_classes))
  lim = X.shape[0]
  Xb_shape = list(X.shape)
  Xb_shape[0] = batch_size
  Xb = np.empty(Xb_shape, dtype=X.dtype)
  yb_shape = list(y.shape)
  yb_shape[0] = batch_size
  yb = np.empty(yb_shape, dtype=y.dtype)
  for i in range(batch_size):
    klass = random.choice(avail)
    seen[klass] += 1
    if seen[klass] == per_class:
      avail.remove(klass)
    index = np.random.choice(y_inv[klass])
    if augment is None:
      Xb[i] = X[index]
    else:
      Xb[i] = augment(X[index])
    yb[i] = y[index]
  return (Xb, yb)
