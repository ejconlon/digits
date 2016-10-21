import os
import sys

import tensorflow as tf

from .data import Env

def main():
  env = Env('.')
  env.assert_ready()
  op = sys.argv[1]
  if op == "inspect":
    model = sys.argv[2]
    inspect(env, model)
  else:
    raise Exception("Unknown op", op)

def inspect(env, name):
  print("inspecting " + name)
  model_path = os.path.join(env.logs, name, 'model.ckpt')
  reader = tf.train.NewCheckpointReader(model_path)
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map:
    print("tensor_name: ", key)
    print(reader.get_tensor(key))

if __name__ == '__main__':
  main()