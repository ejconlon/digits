import random

import numpy as np
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
def rando(img, s=DEFAULT_S, r=DEFAULT_R, t=DEFAULT_T, i=DEFAULT_I):
    scale = random.uniform(1/s, s)
    rot = random.uniform(-r, r)
    trans = (random.uniform(-t, t), random.uniform(-t, t))
    tform = skimage.transform.SimilarityTransform(scale=scale, rotation=rot, translation=trans)
    warped = skimage.transform.warp(img, tform, mode='symmetric')
    if random.random() < i:
      warped = 1.0 - warped
    return warped
