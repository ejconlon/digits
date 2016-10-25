import argparse
import json
import os
import pprint
import sys

import tensorflow as tf

from .data import Env, Loader
from .classifiers import run_train_model, run_test_model
from .metrics import Metrics, read_report, write_report, pickle_to, unpickle_from

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
  train_parser.add_argument('--valid-data', required=True)
  test_parser = subparsers.add_parser('test')
  test_parser.add_argument('--model', required=True)
  test_parser.add_argument('--variant')
  test_parser.add_argument('--test-data', required=True)
  report_parser = subparsers.add_parser('report')
  report_parser.add_argument('--model', required=True)
  report_parser.add_argument('--variant')
  report_parser.add_argument('--role', required=True)
  return parser

def inspect(env, loader, args):
  print('inspecting', args.model)
  model_path = os.path.join(env.logs, args.model, 'model.ckpt')
  reader = tf.train.NewCheckpointReader(model_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map:
    print('tensor_name:', key)
    print(reader.get_tensor(key))

def write_args(env, args):
  args_file = env.resolve_model_file(args.model, args.variant, args.op + '.json', clean=True)
  with open(args_file, 'w') as f:
    json.dump(vars(args), f, sort_keys=True, indent=2)

def write_results(env, args, role, orig, proc, pred):
  report_file = env.resolve_role_file(args.model, args.variant, role, 'report.json', clean=True)
  metrics_file = env.resolve_role_file(args.model, args.variant, role, 'metrics.pickle', clean=True)
  viz_file = env.resolve_role_file(args.model, args.variant, role, 'viz.pickle', clean=True)
  metrics = Metrics(10, pred, proc.y)
  pickle_to(metrics, metrics_file)
  report = metrics.report()
  write_report(report, report_file)
  viz = metrics.viz(orig, proc, (32, 32), 10)
  pickle_to(viz, viz_file)

def train(env, loader, args):
  train_orig, train_proc = loader.load_data(args.train_data)
  valid_orig, valid_proc = loader.load_data(args.valid_data)
  train_pred, valid_pred = run_train_model(env, args.model, args.variant, train_proc, valid_proc)
  write_results(env, args, 'train', train_orig, train_proc, train_pred)
  write_results(env, args, 'valid', valid_orig, valid_proc, valid_pred)

def test(env, loader, args):
  test_orig, test_proc = loader.load_data(args.test_data)
  test_pred = run_test_model(env, args.model, args.variant, test_proc)
  write_results(env, args, 'test', test_orig, test_proc, test_pred)

def report(env, loader, args):
  filename = env.resolve_role_file(args.model, args.variant, args.role, 'report.json')
  report = read_report(filename)
  pprint.pprint(report._asdict())

OPS = {
  'inspect': inspect,
  'train': train,
  'test': test,
  'report': report
}

def main():
  env = Env('.')
  env.assert_ready()
  loader = Loader.from_env(env)
  loader.assert_ready()
  parser = make_parser()
  args = parser.parse_args()
  write_args(env, args)
  OPS[args.op](env, loader, args)

if __name__ == '__main__':
  main()
