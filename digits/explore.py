from base64 import b64encode
from collections import namedtuple
from io import BytesIO, StringIO
import os
import warnings

from IPython.core.display import HTML, display
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
  'viz'
])

def explore(env, model, variant, role):
  report_file = env.resolve_role_file(model, variant, role, 'report.json')
  metrics_file = env.resolve_role_file(model, variant, role, 'metrics.pickle')
  viz_file = env.resolve_role_file(model, variant, role, 'viz.pickle')
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
  return Explorer(report=report, metrics=metrics, viz=viz)

# show image or array of them in ipython
def img_show(arr):
  img_effect(lambda x: display(HTML(img_tag(x))), arr)  

# Given a single image, return a tag
def img_tag(arr):
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
  img = Image.fromarray(arr, mode)
  out = BytesIO()
  img.save(out, format='png')
  return "<img src='data:image/png;base64,{0}'/>".format(b64encode(out.getvalue()).decode('utf-8'))

def viz_table(tab):
  # need to disable truncation for this function because it will chop image tags :(
  old_width = pd.get_option('display.max_colwidth')
  pd.set_option('display.max_colwidth', -1)
  formatters = {
    'proc_image': lambda arr: img_tag(arr)
  }
  buf = StringIO()
  tab.to_html(buf, formatters=formatters, escape=False)
  pd.set_option('display.max_colwidth', old_width)
  return buf.getvalue()
