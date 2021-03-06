"""
Entry point for digits via the command line.

DON'T READ THIS FILE. It's a pile of hacks just to get the pipeline together.

INSTEAD: Run it with `python -m digits.main -h` and poke around the command line options.

OR: Run it with `make run ARGS="-h"`
"""

import argparse
import csv
import gc
import json
import os
import pprint
import shutil
import sys
import tempfile
import warnings

try:
  # Py 3
  from urllib.request import urlopen
except ImportError:
  # Py 2
  from urllib2 import urlopen

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_mldata
import tensorflow as tf

# These are noisy with deprecations that I don't really care about here
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  from runipy.notebook_runner import NotebookRunner
  from IPython.nbformat.current import read, write
  import nbconvert

from .data import Env, Loader, RandomStateContext
from .classifiers import run_train_model, run_test_model, MODELS, TFModel
from .metrics import Metrics, read_report, write_report, pickle_to, unpickle_from
from .params import PARAMS, SEARCH, CONFIGS, find_search_size, has_search_size
from .explore import explore, plot_learning, plot_weights, plot_images, plot_images_array

def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='op')
  inspect_parser = subparsers.add_parser('inspect', help='dump tensorflow parameters for a trained model')
  inspect_parser.add_argument('--model', required=True, help='model type (baseline, tf, vote)')
  inspect_parser.add_argument('--variant', help='unique identifier')
  train_parser = subparsers.add_parser('train', help='train a model')
  train_parser.add_argument('--model', required=True, help='model type (baseline, tf, vote)')
  train_parser.add_argument('--variant', help='unique identifier')
  train_parser.add_argument('--train-data', required=True, help='dataset identifier (mnist-train, crop-small-test, etc)')
  train_parser.add_argument('--valid-data', help='dataset identifier (mnist-train, crop-small-test, etc)')
  train_parser.add_argument('--test-data', help='dataset identifier (mnist-train, crop-small-test, etc)')
  train_parser.add_argument('--preprocessor', help='data preprocessor (noop, gray, color, hog, etc)')
  train_parser.add_argument('--param-set', required=True, help='parameter set identifier (mnist, crop, etc)')
  train_parser.add_argument('--search-set', help='search set identifier')
  train_parser.add_argument('--search-size', type=int, help='number of search iterations')
  train_parser.add_argument('--search-default', type=bool, default=True, help='try default params?')
  train_parser.add_argument('--max-acc', type=float, help='break early acc limit')
  train_parser.add_argument('--random-state', type=int, help='seed for data randomization')
  test_parser = subparsers.add_parser('test', help='test a trained model')
  test_parser.add_argument('--model', required=True, help='model type (baseline, tf, vote)')
  test_parser.add_argument('--variant', help='unique identifier')
  test_parser.add_argument('--test-data', required=True, help='dataset identifier (mnist-train, crop-small-test, etc)')
  test_parser.add_argument('--preprocessor', help='data preprocessor (noop, gray, color, hog, etc)')
  test_parser.add_argument('--param-set', required=True, help='parameter set identifier (mnist, crop, etc)')
  test_parser.add_argument('--random-state', type=int, help='seed for data randomization')
  drive_parser = subparsers.add_parser('drive', help='run a specific config')
  drive_parser.add_argument('--model', required=True, help='model type (baseline, tf, vote)')
  drive_parser.add_argument('--variant', help='unique identifier')
  drive_parser.add_argument('--random-state', type=int, help='seed for data randomization')
  drive_train_parser = drive_parser.add_mutually_exclusive_group(required=False)
  drive_train_parser.add_argument('--train', dest='train', action='store_true', help='train first')
  drive_train_parser.add_argument('--no-train', dest='train', action='store_false', help='skip training')
  drive_parser.set_defaults(train=True)
  drive_parser.add_argument('--max-acc', type=float, help='break early acc limit')
  report_parser = subparsers.add_parser('report', help='show results of trained model')
  report_parser.add_argument('--model', required=True, help='model type (baseline, tf, vote)')
  report_parser.add_argument('--variant', help='unique identifier')
  report_parser.add_argument('--role', required=True, help='role type (train, valid, test)')
  curve_parser = subparsers.add_parser('curve', help='show learning curve of trained model')
  curve_parser.add_argument('--model', required=True, help='model type (baseline, tf, vote)')
  curve_parser.add_argument('--variant', help='unique identifier')
  params_parser = subparsers.add_parser('params', help='show params of trained model')
  params_parser.add_argument('--model', required=True, help='model type (baseline, tf, vote)')
  params_parser.add_argument('--variant', help='unique identifier')
  summarize_parser = subparsers.add_parser('summarize', help='describe a dataset')
  summarize_parser.add_argument('--data', required=True, help='dataset identifier (mnist-train, crop-small-test, etc)')
  subparsers.add_parser('fetch_mnist', help='fetch MNIST dataset')
  subparsers.add_parser('fetch_svhn', help='fetch SVHN dataset')
  subparsers.add_parser('notebooks', help='render all notebooks to results')
  return parser

def inspect(env, loader, args):
  print('inspecting', args.model, args.variant)
  model_path = os.path.join('logs', env.model_name_plus(args.model, args.variant), 'model.ckpt')
  reader = tf.train.NewCheckpointReader(model_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map:
    print('tensor_name:', key)
    print(reader.get_tensor(key))

def write_results(env, model, variant, role, proc, metrics, activations, viz):
  num_act_examples = 1
  report_file = env.resolve_role_file(model, variant, role, 'report.json', clean=True)
  metrics_file = env.resolve_role_file(model, variant, role, 'metrics.pickle', clean=True)
  viz_file = env.resolve_role_file(model, variant, role, 'viz.pickle', clean=True)
  activations_file = env.resolve_role_file(model, variant, role, 'activations.pickle', clean=True)
  pickle_to(metrics, metrics_file)
  report = metrics.report()
  write_report(report, report_file)
  pickle_to(viz, viz_file)
  if activations is not None:
    pickle_to(activations, activations_file)
  print(env.model_name_plus(model, variant), '/', role)
  print('accuracy', metrics.accuracy())
  metrics.print_classification_report()
  # Now plot stuff!
  e = explore(env, model, variant, role, assert_complete=False)
  if role == 'valid':
    if e.learning_curve is not None:
      curve_file = env.resolve_model_file(model, variant, 'learning_curve.png', clean=True)
      plot_learning(e.learning_curve, dest=curve_file)
    if e.conv_weights is not None:
      weights_file = env.resolve_model_file(model, variant, 'weights_0.png', clean=True)
      plot_weights(e.conv_weights, 0, dest=weights_file)
  if model == 'tf':
    viz_dict = e.viz._asdict()
    for target in ['correct_certain', 'wrong_certain', 'correct_uncertain', 'wrong_uncertain']:
      out_file = env.resolve_role_file(model, variant, role, target + '_images.png', clean=True)
      # The to_dict in the lambda here is to get around pandas indexing with python 2
      # that expects one dimensional array cells
      plot_images(viz_dict[target], lambda r: '(%d, %d, %.2f)' % (r.gold_class, r.pred_class, r.p), lambda r: r.to_dict()['proc_image'], dest=out_file)
      if len(activations[target]) > 0:
        f = activations[target]
        for i in range(num_act_examples):
          act_img_out = env.resolve_role_file(model, variant, role, target + '_activations_' + str(i) + '.png', clean=True)
          x = f[(f.layer == 0) & (f.image == i)]
          if len(x) > 0:
            plot_images(x, lambda r: None, lambda r: r.to_dict()['activations'], dest=act_img_out)

def write_params(env, model, variant, params):
  params_file = env.resolve_model_file(model, variant, 'params.json', clean=True)
  with open(params_file, 'w') as f:
    json.dump(vars(params), f, sort_keys=True, indent=2)

def run_activations(env, model, variant, proc, metrics):
  viz = metrics.viz(proc, 10)
  if model == 'tf':
    viz_dict = viz._asdict()
    activations = {}
    model = TFModel(env, model, variant)
    columns = ['layer', 'image', 'channel', 'activations']
    for target in ['correct_certain', 'wrong_certain', 'correct_uncertain', 'wrong_uncertain']:
      vs = viz_dict[target].proc_image.values
      X = np.array([vs[i] for i in range(len(vs))])
      acts = model.activations(X)
      recs = []
      for layer in range(len(acts)):
        for index in range(acts[layer].shape[0]):
          for channel in range(acts[layer].shape[1]):
            a = acts[layer][index][channel]
            recs.append({ 'layer': layer, 'image': index, 'channel': channel, 'activations': a })
      activations[target] = pd.DataFrame.from_records(recs, columns=columns)
  else:
    activations = None
  return (activations, viz)

def run_train(env, loader, args):
  assert args.model in MODELS
  assert args.model in PARAMS
  assert args.param_set in PARAMS[args.model]
  train_proc = loader.load_data(args.train_data, args.preprocessor, args.random_state)
  if args.valid_data is not None:
    valid_proc = loader.load_data(args.valid_data, args.preprocessor, args.random_state)
  else:
    valid_proc = None
  if args.test_data is not None:
    test_proc = loader.load_data(args.test_data, args.preprocessor, args.random_state)
  else:
    test_proc = None

  best_valid_acc = None
  best_variant = None
  original_acc = None

  search_size = args.search_size
  if args.search_set is not None:
    if search_size is None:
      search_size = find_search_size(args.model, args.search_set)
    else:
      assert not has_search_size(args.model, args.search_set)

  if args.search_set is None or args.search_default:
    gc.collect()
    print('running default variant')
    final_params, valid_metrics = run_train_model(env, args.model, args.variant, train_proc, valid_proc, args.param_set, max_acc=args.max_acc)
    write_params(env, args.model, args.variant, final_params)
    if valid_metrics is not None:
      valid_activations, valid_viz = run_activations(env, args.model, args.variant, valid_proc, valid_metrics)
      write_results(env, args.model, args.variant, 'valid', valid_proc, valid_metrics, valid_activations, valid_viz)
      original_acc = valid_metrics.accuracy()
      best_valid_acc = original_acc
      best_variant = args.variant
  else:
    print('skipping default variant')

  if args.search_set is not None:
    assert search_size > 0
    assert valid_proc is not None
    for i in range(search_size):
      gc.collect()
      variant_i = args.variant + '__' + str(i)
      cand_params, cand_valid_metrics = \
        run_train_model(env, args.model, variant_i, train_proc, valid_proc, args.param_set, args.search_set, i, max_acc=args.max_acc)
      write_params(env, args.model, variant_i, cand_params)
      valid_activations, valid_viz = run_activations(env, args.model, variant_i, valid_proc, cand_valid_metrics)
      write_results(env, args.model, variant_i, 'valid', valid_proc, cand_valid_metrics, valid_activations, valid_viz)
      cand_valid_acc = cand_valid_metrics.accuracy()
      if best_valid_acc is None or cand_valid_acc > best_valid_acc:
        print('Better variant {} accuracy {}'.format(variant_i, cand_valid_acc))
        best_variant = variant_i
        best_valid_acc = cand_valid_acc
        final_params = cand_params
        valid_metrics = cand_valid_metrics
    assert best_variant is not None
    if best_variant == args.variant:
      print('Best variant is default variant, {}'.format(original_acc))
    else:
      print('Best variant {}, {} > {}'.format(best_variant, best_valid_acc, original_acc)) 
      src_path = env.resolve_model(args.model, variant_i)
      dest_path = env.resolve_model(args.model, args.variant)
      shutil.rmtree(dest_path)
      shutil.copytree(src_path, dest_path)
      pprint.pprint(final_params)

  if args.test_data is not None:
    gc.collect()
    test_metrics = run_test_model(env, args.model, args.variant, test_proc, args.param_set)
    test_activations, test_viz = run_activations(env, args.model, args.variant, test_proc, test_metrics)
    write_results(env, args.model, args.variant, 'test', test_proc, test_metrics, test_activations, test_viz)

def run_test(env, loader, args):
  assert args.model in MODELS
  assert args.model in PARAMS
  assert args.param_set in PARAMS[args.model]
  test_proc = loader.load_data(args.test_data, args.preprocessor, args.random_state)
  test_metrics = run_test_model(env, args.model, args.variant, test_proc, args.param_set)
  test_activations, test_viz = run_activations(env, args.model, args.variant, test_proc, test_metrics)
  write_results(env, args.model, args.variant, 'test', test_proc, test_metrics, test_activations, test_viz)

def report(env, loader, args):
  filename = env.resolve_role_file(args.model, args.variant, args.role, 'report.json')
  report = read_report(filename)
  pprint.pprint(report._asdict())

def curve(env, loader, args):
  filename = env.resolve_model_file(args.model, args.variant, 'learning_curve.csv')
  curve = pd.read_csv(filename)
  print(curve)

def params(env, loader, args):
  filename = env.resolve_model_file(args.model, args.variant, 'params.json')
  with open(filename, 'r') as f:
    print(f.read())

def summarize(env, loader, args):
  proc = loader.load_data(args.data, preprocessor='noop', random_state=None)
  print('proc X', proc.X.shape)
  print('proc y', proc.y.shape)

def fetch_mnist(env, loader, args):
  data_home = env.resolve('data')
  fetch_mldata('MNIST original', data_home=data_home)

def fetch_svhn(env, loader, args):
  for role in ['train', 'test', 'extra']:
    filename = role + '_32x32.mat'
    path = os.path.join(loader.data_path, filename)
    if os.path.isfile(path):
      print('found svhn', role)
    else:
      print('fetching svhn', role)
      url = 'http://ufldl.stanford.edu/housenumbers/' + filename
      response = urlopen(url)
      with open(path, 'wb') as f:
        f.write(response.read())

class Cwd:
  def __init__(self, d):
    self.d = d
    self.c = None

  def __enter__(self):
    self.c = os.getcwd()
    os.chdir(self.d)

  def __exit__(self, x, y, z):
    os.chdir(self.c)
    self.c = None

def notebooks(env, loader, args):
  nb_path = env.resolve('notebooks')
  res_path = env.resolve('results')
  for filename in os.listdir(nb_path):
    if filename.endswith('.ipynb'):
      nb_name = filename.split('.')[0]
      print('running', nb_name)
      src_path = os.path.join(nb_path, filename)
      html_path = os.path.join(res_path, nb_name + '.html')
      with open(src_path, 'r') as f:
        notebook = read(f, 'json')
        r = NotebookRunner(notebook, working_dir=nb_path)
        r.run_notebook()
        dest_path = tempfile.mkstemp()
        t = tempfile.NamedTemporaryFile()
        with open(t.name, 'w') as g:
          write(r.nb, g)
        exporter = nbconvert.HTMLExporter()
        body, resources = exporter.from_filename(t.name)
        with open(html_path, 'wb') as g:
          g.write(body.encode('utf-8'))

def run_model(
  env, loader, model, variant, train_data_name, valid_data_name, test_data_name,
  preprocessor, param_set, max_acc, train=True, search_set=None, search_size=None, search_default=True, check_ser=False, random_state=None):

  train_args = argparse.Namespace(
    random_state=random_state,
    op='train',
    model=model,
    variant=variant,
    train_data=train_data_name,
    valid_data=valid_data_name,
    test_data=test_data_name,
    preprocessor=preprocessor,
    param_set=param_set,
    search_set=search_set,
    search_size=search_size,
    search_default=search_default,
    max_acc=max_acc
  )

  test_args = argparse.Namespace(
    random_state=random_state,
    op='test',
    model=model,
    variant=variant,
    test_data=test_data_name,
    preprocessor=preprocessor,
    param_set=param_set
  )

  if train:
    sub_main(env, loader, train_args)
    if check_ser and test_data_name is not None:
      test_metrics1 = unpickle_from(env.resolve_role_file(model, variant, 'test', 'metrics.pickle'))
      sub_main(env, loader, test_args)
      test_metrics2 = unpickle_from(env.resolve_role_file(model, variant, 'test', 'metrics.pickle'))
      # Sanity check, they should have run on the same dataset
      np.testing.assert_array_equal(test_metrics1.gold, test_metrics2.gold)
      # Now check that we've correctly predicted with the serialized model
      np.testing.assert_array_equal(test_metrics1.pred, test_metrics2.pred)
  else:
    sub_main(env, loader, test_args)

def drive(env, loader, args):
  config = next(c for c in CONFIGS if c.model == args.model and c.variant == args.variant)
  run_model(env, loader, max_acc=args.max_acc, train=args.train,
    random_state=args.random_state, **config.__dict__)
  
OPS = {
  'inspect': inspect,
  'train': run_train,
  'test': run_test,
  'report': report,
  'curve': curve,
  'params': params,
  'summarize': summarize,
  'fetch_mnist': fetch_mnist,
  'fetch_svhn': fetch_svhn,
  'notebooks': notebooks,
  'drive': drive
}

def sub_main(env, loader, args):
  OPS[args.op](env, loader, args)

def main():
  env = Env('.')
  env.assert_ready()
  loader = Loader.from_env(env)
  loader.assert_ready()
  parser = make_parser()
  args = parser.parse_args()
  assert args.op is not None
  sub_main(env, loader, args)

if __name__ == '__main__':
  main()
