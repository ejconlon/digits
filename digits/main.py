import argparse
import os
import sys

import tensorflow as tf

from .data import Env

def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='op')
  inspect_parser = subparsers.add_parser('inspect')
  inspect_parser.add_argument('--model', required=True)
  train_parser = subparsers.add_parser('train')
  train_parser.add_argument('--model', required=True)
  train_parser.add_argument('--train-data', required=True)
  train_parser.add_argument('--valid-data', required=False)
  train_parser.add_argument('--test-data', required=False)
  test_parser = subparsers.add_parser('test')
  test_parser.add_argument('--model', required=True)
  test_parser.add_argument('--test-data', required=True)
  return parser

def inspect(env, args):
  print('inspecting', args.model)
  model_path = os.path.join(env.logs, args.model, 'model.ckpt')
  reader = tf.train.NewCheckpointReader(model_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map:
    print('tensor_name:', key)
    print(reader.get_tensor(key))

def train(env, args):
  raise Exception("TODO")

def test(env, args):
  raise Exception("TODO")

ops = {
  'inspect': inspect,
  'train': train,
  'test': test
}

def main():
  env = Env('.')
  env.assert_ready()
  parser = make_parser()
  args = parser.parse_args()
  ops[args.op](env, args)

if __name__ == '__main__':
  main()
