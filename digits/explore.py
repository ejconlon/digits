from base64 import b64encode
from collections import namedtuple
from io import BytesIO, StringIO
import json
import math
import os
import warnings

from IPython.core.display import HTML, display
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import skimage
import skimage.exposure

from .metrics import read_report, unpickle_from
from .images import img_effect

Explorer = namedtuple('Explorer', [
  'report',
  'metrics',
  'viz',
  'learning_curve',
  'params',
  'conv_weights',
  'activations'
])

def plot_weights(weight_frame, layer, show=False, dest=None):
  assert show or dest is not None
  plt.clf()

  frame_layer = weight_frame[weight_frame.layer == layer]
  size = int(math.ceil(math.sqrt(len(frame_layer))))
  fig, axes = plt.subplots(size, size, subplot_kw={'xticks': [], 'yticks': []})
  i = 0
  for ax in axes.flat:
    if i < len(frame_layer):
      fi = frame_layer.iloc[i]
      assert fi.layer == layer
      ws = fi.to_dict()['weights']
      x, _ = img_fudge(ws)
      ax.imshow(x, interpolation='nearest', cmap='seismic')
    else:
      ax.set_visible(False)
    i += 1

  if show:
    plt.show()

  if dest is not None:
    plt.savefig(dest, bbox_inches='tight')

def plot_learning(curve, show=False, dest=None):
  assert show or dest is not None
  plt.clf()

  fig, ax1 = plt.subplots()
  ax1.set_ylabel('acc')
  ax1.set_autoscaley_on(False)
  ax1.set_ylim([0, 1])
  l1 = ax1.plot(curve.seen, curve.train_acc, label='train acc')
  l2 = ax1.plot(curve.seen, curve.valid_acc, label='valid acc')

  ax2 = ax1.twinx()
  max_loss = max(np.max(curve.train_loss), np.max(curve.valid_loss))
  ax2.set_ylabel('loss')
  ax2.set_ylim([0, max_loss])
  l3 = ax2.plot(curve.seen, curve.train_loss, linestyle='--', label='train loss')
  l4 = ax2.plot(curve.seen, curve.valid_loss, linestyle='--', label='valid loss')

  ls = l1 + l2 + l3 + l4
  labs = [l.get_label() for l in ls]
  ax1.legend(ls, labs, loc=0)

  if show:
    plt.show()

  if dest is not None:
    plt.savefig(dest, bbox_inches='tight')

def explore(env, model, variant, role, assert_complete=False):
  report_file = env.resolve_role_file(model, variant, role, 'report.json')
  metrics_file = env.resolve_role_file(model, variant, role, 'metrics.pickle')
  viz_file = env.resolve_role_file(model, variant, role, 'viz.pickle')
  lc_file = env.resolve_model_file(model, variant, 'learning_curve.csv')
  params_file = env.resolve_model_file(model, variant, 'params.json')
  cw_file = env.resolve_model_file(model, variant, 'conv_weights.pickle')
  act_file = env.resolve_role_file(model, variant, role, 'activations.pickle')
  if os.path.isfile(report_file):
    report = read_report(report_file)
  else:
    report = None
  if os.path.isfile(metrics_file):
    metrics = unpickle_from(metrics_file)
  else:
    metrics = None
  if os.path.isfile(viz_file):
    viz = unpickle_from(viz_file)
  else:
    viz = None
  if os.path.isfile(lc_file):
    learning_curve = pd.read_csv(lc_file)
  else:
    learning_curve = None
  if os.path.isfile(params_file):
    with open(params_file, 'r') as f:
      params = json.load(f)
  else:
    params = None
  if os.path.isfile(cw_file):
    conv_weights = unpickle_from(cw_file)
  else:
    conv_weights = None
  if os.path.isfile(act_file):
    activations = unpickle_from(act_file)
  else:
    activations = None
  if assert_complete:
    assert report is not None
    assert metrics is not None
    assert viz is not None
    assert params is not None
    if model == 'tf':
      assert learning_curve is not None
      assert conv_weights is not None
      # TODO enable
      #assert activations is not None
  return Explorer(report, metrics, viz, learning_curve, params, conv_weights, activations)

# show image or array of them in ipython
def img_show(arr):
  img_effect(lambda x: display(HTML(img_tag(x))), arr)  

# make gray images look gray to matplotlib by removing the dummy depth dim
def img_fudge(img):
  if len(img.shape) == 2:
    return (img, True)
  elif len(img.shape) == 3 and img.shape[2] == 1:
    return (img.reshape((img.shape[0], img.shape[1])), True)
  else:
    return (img, False)

def img_obj(arr):
  if arr.dtype != np.uint8:
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      arr = skimage.exposure.rescale_intensity(arr, (0, 1.0))
      arr = skimage.img_as_ubyte(arr)
  if len(arr.shape) == 2:
    mode = 'L'
  elif len(arr.shape) == 3:
    if arr.shape[2] == 3:
      mode = 'RGB'
    elif arr.shape[2] == 1:
      mode = 'L'
      arr = arr.reshape(arr.shape[:2])
    else:
      raise Exception('Invalid depth', arr.shape[2])
  else:
    raise Exception('Invalid shape', arr.shape)
  return Image.fromarray(arr, mode)

# Given a single image, return a tag
def img_tag(arr):
  img = img_obj(arr)
  out = BytesIO()
  img.save(out, format='png')
  return "<img src='data:image/png;base64,{0}'/>".format(b64encode(out.getvalue()).decode('utf-8'))

def viz_table(tab):
  # need to disable truncation for this function because it will chop image tags :(
  old_width = pd.get_option('display.max_colwidth')
  pd.set_option('display.max_colwidth', -1)
  formatters = {
    'proc_image': lambda arr: img_tag(arr),
    'weights': lambda arr: img_tag(arr)
  }
  buf = StringIO()
  tab.to_html(buf, formatters=formatters, escape=False)
  pd.set_option('display.max_colwidth', old_width)
  return buf.getvalue()

def plot_images(frame, titler, imager, rows=None, cols=None, show=False, dest=None):
  assert show or dest is not None
  plt.clf()

  if rows is None:
    assert cols is None
    rows = int(math.ceil(math.sqrt(len(frame))))
    cols = rows
  else:
    assert cols is not None
  
  fig, axes = plt.subplots(rows, cols, subplot_kw={'xticks': [], 'yticks': []})

  fig.subplots_adjust(hspace=0.5, wspace=0.2)

  i = 0
  for ax in axes.flat:
    if i < len(frame):
      row = frame.iloc[i]
      title = titler(row)
      img, is_gray = img_fudge(imager(row))
      if title is not None:
        ax.set_title(title)
      if is_gray:
        cmap = 'gray'
      else:
        cmap = None
      ax.imshow(img, cmap=cmap, interpolation='nearest')
    else:
      ax.set_visible(False)
    i += 1

  if show:
    plt.show()

  if dest is not None:
    plt.savefig(dest, bbox_inches='tight')

def plot_images_array(arr, rows=None, cols=None, show=False, dest=None):
  assert show or dest is not None
  plt.clf()

  if rows is None:
    assert cols is None
    rows = int(math.ceil(math.sqrt(arr.shape[0])))
    cols = rows
  else:
    assert cols is not None
  
  fig, axes = plt.subplots(rows, cols, subplot_kw={'xticks': [], 'yticks': []})

  fig.subplots_adjust(hspace=0.5, wspace=0.2)

  i = 0
  for ax in axes.flat:
    if i < arr.shape[0]:
      img, is_gray = img_fudge(arr[i])
      if is_gray:
        cmap = 'gray'
      else:
        cmap = None
      ax.imshow(img, cmap=cmap, interpolation='nearest')
    else:
      ax.set_visible(False)
    i += 1

  if show:
    plt.show()

  if dest is not None:
    plt.savefig(dest, bbox_inches='tight')
