import argparse
import csv
import json
import os
import pprint
import shutil
import sys
import tempfile
import urllib.request
import warnings

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
from .classifiers import run_train_model, run_test_model, MODELS
from .metrics import Metrics, read_report, write_report, pickle_to, unpickle_from
from .params import PARAMS, SEARCH, CONFIGS

def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='op')
  inspect_parser = subparsers.add_parser('inspect')
  inspect_parser.add_argument('--model', required=True)
  inspect_parser.add_argument('--variant')
  train_parser = subparsers.add_parser('train')
  train_parser.add_argument('--model', required=True)
  train_parser.add_argument('--variant')
  train_parser.add_argument('--train-data', required=True)
  train_parser.add_argument('--valid-data')
  train_parser.add_argument('--test-data')
  train_parser.add_argument('--preprocessor')
  train_parser.add_argument('--param-set', required=True)
  train_parser.add_argument('--search-set')
  train_parser.add_argument('--search-size', type=int)
  train_parser.add_argument('--random-state', type=int)
  test_parser = subparsers.add_parser('test')
  test_parser.add_argument('--model', required=True)
  test_parser.add_argument('--variant')
  test_parser.add_argument('--test-data', required=True)
  test_parser.add_argument('--preprocessor')
  test_parser.add_argument('--param-set', required=True)
  test_parser.add_argument('--random-state', type=int)
  drive_parser = subparsers.add_parser('drive')
  drive_parser.add_argument('--model', required=True)
  drive_parser.add_argument('--variant')
  drive_parser.add_argument('--random-state', type=int)
  report_parser = subparsers.add_parser('report')
  report_parser.add_argument('--model', required=True)
  report_parser.add_argument('--variant')
  report_parser.add_argument('--role', required=True)
  curve_parser = subparsers.add_parser('curve')
  curve_parser.add_argument('--model', required=True)
  curve_parser.add_argument('--variant')
  params_parser = subparsers.add_parser('params')
  params_parser.add_argument('--model', required=True)
  params_parser.add_argument('--variant')
  summarize_parser = subparsers.add_parser('summarize')
  summarize_parser.add_argument('--data', required=True)
  subparsers.add_parser('fetch_mnist')
  subparsers.add_parser('fetch_svhn')
  subparsers.add_parser('notebooks')
  return parser

def inspect(env, loader, args):
  print('inspecting', args.model)
  model_path = os.path.join(env.logs, args.model, 'model.ckpt')
  reader = tf.train.NewCheckpointReader(model_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map:
    print('tensor_name:', key)
    print(reader.get_tensor(key))

def write_results(env, model, variant, role, proc, metrics):
  report_file = env.resolve_role_file(model, variant, role, 'report.json', clean=True)
  metrics_file = env.resolve_role_file(model, variant, role, 'metrics.pickle', clean=True)
  viz_file = env.resolve_role_file(model, variant, role, 'viz.pickle', clean=True)
  pickle_to(metrics, metrics_file)
  report = metrics.report()
  write_report(report, report_file)
  viz = metrics.viz(proc, 10)
  pickle_to(viz, viz_file)
  print(env.model_name_plus(model, variant), '/', role)
  print('accuracy', metrics.accuracy())
  metrics.print_classification_report()

def write_params(env, model, variant, params):
  params_file = env.resolve_model_file(model, variant, 'params.json', clean=True)
  with open(params_file, 'w') as f:
    json.dump(vars(params), f, sort_keys=True, indent=2)

def run_train(env, loader, args):
  assert args.model in MODELS
  assert args.model in PARAMS
  assert args.param_set in PARAMS[args.model]
  _, train_proc = loader.load_data(args.train_data, args.preprocessor, args.random_state)
  if args.valid_data is not None:
    _, valid_proc = loader.load_data(args.valid_data, args.preprocessor, args.random_state)
  else:
    valid_proc = None
  if args.test_data is not None:
    _, test_proc = loader.load_data(args.test_data, args.preprocessor, args.random_state)
  else:
    test_proc = None

  best_valid_acc = None
  best_variant = None
  original_acc = None

  print('running default variant')
  final_params, valid_metrics = run_train_model(env, args.model, args.variant, train_proc, valid_proc, args.param_set)
  if valid_metrics is not None:
    write_results(env, args.model, args.variant, 'valid', valid_proc, valid_metrics)
  write_params(env, args.model, args.variant, final_params)

  if valid_metrics is not None:
    original_acc = valid_metrics.accuracy()
    best_valid_acc = original_acc
    best_variant = args.variant

  if args.search_set is not None:
    assert args.search_size is not None
    assert args.search_size > 0
    assert valid_proc is not None
    for i in range(args.search_size):
      variant_i = args.variant + '__' + str(i)
      cand_params, cand_valid_metrics = \
        run_train_model(env, args.model, variant_i, train_proc, valid_proc, args.param_set, args.search_set)
      write_results(env, args.model, variant_i, 'valid', valid_proc, cand_valid_metrics)
      write_params(env, args.model, variant_i, cand_params)
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
    test_metrics = run_test_model(env, args.model, args.variant, test_proc, args.param_set)
    write_results(env, args.model, args.variant, 'test', test_proc, test_metrics)

def run_test(env, loader, args):
  assert args.model in MODELS
  assert args.model in PARAMS
  assert args.param_set in PARAMS[args.model]
  _, test_proc = loader.load_data(args.test_data, args.preprocessor, args.random_state)
  test_metrics = run_test_model(env, args.model, args.variant, test_proc, args.param_set)
  write_results(env, args.model, args.variant, 'test', test_proc, test_metrics)

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
    pprint.pprint(f.read())

def summarize(env, loader, args):
  orig, proc = loader.load_data(args.data)
  print('orig X', orig.X.shape)
  print('orig y', orig.y.shape)
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
      print('found', role)
    else:
      print('fetching', role)
      url = 'http://ufldl.stanford.edu/housenumbers/' + filename
      urllib.request.urlretrieve(url, path)

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
        with open(html_path, 'w') as g:
          g.write(body)

def run_model(
  env, loader, model, variant, train_data_name, valid_data_name, test_data_name,
  preprocessor, param_set, search_set=None, search_size=None, check_ser=False, random_state=None):
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
    search_size=search_size
  )
  sub_main(env, loader, train_args)

  if check_ser and test_data_name is not None:
    test_metrics1 = unpickle_from(env.resolve_role_file(model, variant, 'test', 'metrics.pickle'))

    test_args = argparse.Namespace(
      random_state=random_state,
      op='test',
      model=model,
      variant=variant,
      test_data=test_data_name,
      preprocessor=preprocessor,
      param_set=param_set
    )
    sub_main(env, loader, test_args)

    test_metrics2 = unpickle_from(env.resolve_role_file(model, variant, 'test', 'metrics.pickle'))

    # Sanity check, they should have run on the same dataset
    np.testing.assert_array_equal(test_metrics1.gold, test_metrics2.gold)

    # Now check that we've correctly predicted with the serialized model
    np.testing.assert_array_equal(test_metrics1.pred, test_metrics2.pred)

def drive(env, loader, args):
  config = next(c for c in CONFIGS if c.model == args.model and c.variant == args.variant)
  run_model(env, loader, random_state=args.random_state, **config.__dict__)
        
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
