import argparse
import os
import pprint
import sys

import tensorflow as tf

from .data import Env, Loader
from .classifiers import train_model, test_model, load_report
from .metrics import read_report

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

def train(env, loader, args):
  train_data = loader.load_data(args.train_data)
  valid_data = loader.load_data(args.valid_data)
  run_train_model(env, args.model, train_data, valid_data)

def test(env, loader, args):
  test_data = loader.load_data(args.test_data)
  run_test_model(env, args.model, test_data)

def report(env, loader, args):
  filename = os.path.join(env.path, 'logs', args.model, args.role, 'report.json')
  report = run_read_report(filename)
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
  OPS[args.op](env, loader, args)

if __name__ == '__main__':
  main()
