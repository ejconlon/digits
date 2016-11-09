from argparse import Namespace

PARAMS = {
  'baseline': {
    'mnist': Namespace(
      num_classes = 10
    ),
    'crop': Namespace(
      num_classes = 10
    )
  },
  'tf': {
    'mnist': Namespace(
      num_classes = 10,
      # regularization param 0.0001 for mnist, 0.00000001 for crop?
      lam =  0.001,
      # learning rate
      alpha = 0.003,
      # decay alpha by this every n steps
      decay_factor = 0.66,
      # update alpha after this number of steps
      decay_step = 100,
      # TODO 150k for mnist
      training_iters = 150000,  
      # number of examples per descent
      batch_size = 128,
      # number of examples per display step
      display_size = 512,
      # number of batches per display/validation step
      display_step = 10,
      # keep_prob, 1.0 keep all
      dropout = 0.85,
      # (width, depth) of convolutional layers
      convs = [(5, 32), (5, 64)],
      # size of fully connected layers
      fcs = [1024],
      # randomize image rotation, etc
      use_rando = True
    ),
    'crop': Namespace(
      num_classes = 10,
      lam =  0.000001,
      alpha = 0.0001,
      training_iters = 100000,
      batch_size = 128,
      display_size = 512,
      display_step = 10,
      dropout = 0.75,
      convs = [(5, 32), (3, 32), (3, 32)],
      fcs = [512],
      use_rando = False
    )
  }
}

SEARCH = {
  'tf': {
    'mnist': Namespace(
      use_rando = [True, False],
      lam = [0.0001, 0.0003, 0.001, 0.00003],
      alpha = [0.001, 0.003, 0.0001],
      fcs = [[1024], [512]],
      convs = [[(5, 32), (5, 64)], [(5, 64), (5, 128)]],
      dropout = [.65, .75, .85]
    ),
    'mnist2': [
      Namespace(
        convs = [(5, 64), (5, 128)]
      )
    ]
  }
}

def find_search_size(model, search_set):
  s = SEARCH[model][search_set]
  if type(s) == list:
    return len(s)
  else:
    raise Exception('Search set not a list: ' + model + ' / ' + search_set)

def has_search_size(model, search_set):
  s = SEARCH[model][search_set]
  return type(s) == list

CONFIGS = [
  Namespace(
    model='baseline',
    variant='crop',
    train_data_name='crop-train-small',
    valid_data_name=None,
    test_data_name='crop-test-small',
    preprocessor='flat-gray',
    param_set='crop'
  ),

  Namespace(
    model='baseline',
    variant='mnist',
    train_data_name='mnist-train',
    valid_data_name=None,
    test_data_name='mnist-test',
    preprocessor='flat-gray',
    param_set='mnist'
  ),

  Namespace(
    model='tf',
    variant='crop-huge',
    train_data_name='crop-train-huge',
    valid_data_name=None,
    test_data_name='crop-test-huge',
    preprocessor='color',
    param_set='crop'
  ),

  Namespace(
    model='tf',
    variant='crop-big',
    train_data_name='crop-train-big',
    valid_data_name=None,
    test_data_name='crop-test-big',
    preprocessor='color',
    param_set='crop'
  ),

  Namespace(
    model='tf',
    variant='crop-small',
    train_data_name='crop-train-small',
    valid_data_name=None,
    test_data_name='crop-test-small',
    preprocessor='color',
    param_set='crop'
  ),

  Namespace(
    model='tf',
    variant='mnist',
    train_data_name='mnist-train',
    valid_data_name='mnist-valid',
    test_data_name='mnist-test',
    preprocessor='color',
    param_set='mnist',
    check_ser=True
  ),

  Namespace(
    model='tf',
    variant='mnist-search',
    train_data_name='mnist-train',
    valid_data_name='mnist-valid',
    test_data_name='mnist-test',
    preprocessor='color',
    param_set='mnist',
    search_set='mnist',
    search_size=2
  )
]

def assert_no_dupes():
  s = set()
  for c in CONFIGS:
    k = (c.model, c.variant)
    if k in s:
      raise Exception('Dupe config key: ' + str(k))
    else:
      s.add(k)

assert_no_dupes()
