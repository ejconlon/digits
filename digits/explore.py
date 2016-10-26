from base64 import b64encode
from collections import namedtuple
from io import BytesIO, StringIO
import os

import pandas as pd
from PIL import Image

from .metrics import read_report, unpickle_from

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

def img_tag(arr, mode=None):
  img = Image.fromarray(arr, 'RGB')
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
