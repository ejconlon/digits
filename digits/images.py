import random

import numpy as np
import skimage.transform

DEFAULT_S=1.03
DEFAULT_R=0.15
DEFAULT_T=1.5

# is the array a single image or multiple?
def is_single_img(arr):
  return len(arr.shape) == 2 or \
    (len(arr.shape) == 3 and (arr.shape[2] == 1 or arr.shape[2] == 3))

# apply a function intelligently to an image or array of images
def img_map(f, arr):
  if is_single_img(arr):
    return f(arr)
  else:
    return np.concatenate([f(arr[i]) for i in range(arr.shape[0])])

# apply a function for side effects
def img_effect(f, arr):
  if is_single_img(arr):
    f(arr)
  else:
    for i in range(arr.shape[0]):
      f(arr[i])

# apply a random transformation to an image
def rando(img, s=DEFAULT_S, r=DEFAULT_R, t=DEFAULT_T):
    scale = random.uniform(1/s, s)
    rot = random.uniform(-r, r)
    trans = (random.uniform(-t, t), random.uniform(-t, t))
    tform = skimage.transform.SimilarityTransform(scale=scale, rotation=rot, translation=trans)
    return skimage.transform.warp(img, tform, mode='symmetric')
