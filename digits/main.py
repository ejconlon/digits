import argparse
import os
import sys

import tensorflow as tf

from .data import Env

def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='op', help='foo')
  inspect_parser = subparsers.add_parser('inspect', help='bar')
  inspect_parser.add_argument('--model', required=True, help='baz')
  return parser

def main():
  env = Env('.')
  env.assert_ready()
  parser = make_parser()
  args = parser.parse_args()
  if args.op == "inspect":
    inspect(env, args.model)
  else:
    raise Exception("Unknown op", args.op)

def inspect(env, name):
  print('inspecting', name)
  model_path = os.path.join(env.logs, name, 'model.ckpt')
  reader = tf.train.NewCheckpointReader(model_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map:
    print('tensor_name:', key)
    print(reader.get_tensor(key))

if __name__ == '__main__':
  main()