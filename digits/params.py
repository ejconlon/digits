from argparse import Namespace

from .images import DEFAULT_SCALE, DEFAULT_TRANSLATION, DEFAULT_ROTATION, DEFAULT_INVERSION

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
      lam = 1e-4,
      # learning rate
      alpha = 0.001,
      # decay alpha by this every n steps
      decay_factor = 0.1,
      # update alpha after this number of steps
      decay_step = 350,
      # number of display steps to break if validation doesn't improve
      break_display_step = 10,
      # TODO 150k for mnist
      training_iters = 150000,  
      # number of examples per descent
      batch_size = 100,
      # number of examples per display step
      display_size = 200,
      # number of batches per display/validation step
      display_step = 25,
      # keep_prob, 1.0 keep all
      dropout = 0.90,
      # (width, depth) of convolutional layers
      convs = [(5, 64), (5, 64)],
      # size of fully connected layers
      fcs = [1024],
      # randomize image rotation, etc
      use_rando = True,
      rando_scale = DEFAULT_SCALE,
      rando_translation = DEFAULT_TRANSLATION,
      rando_rotation = DEFAULT_ROTATION,
      rando_inversion = DEFAULT_INVERSION,
      invert=False
    ),
    'crop': Namespace(
      num_classes = 10,
      lam = 1e-4,
      alpha = 0.003,
      decay_factor = 0.1,
      decay_step = 350,
      break_display_step = 20,
      training_iters = 1000000,
      batch_size = 200,
      display_size = 1000,
      display_step = 25,
      dropout = 0.90,
      convs = [(5, 64), (5, 64)],
      fcs = [1024],
      use_rando = False,
      rando_scale = DEFAULT_SCALE,
      rando_translation = DEFAULT_TRANSLATION,
      rando_rotation = DEFAULT_ROTATION,
      rando_inversion = DEFAULT_INVERSION,
      invert = True
    ),
  }
}

PARAMS['vote'] = PARAMS['tf']

SEARCH = {
  'tf': {
    'mnist': Namespace(
      use_rando = [True, False],
      lam = [0.0001, 0.0003, 0.001, 0.00003],
      alpha = [0.001, 0.003, 0.0001],
      fcs = [[1024], [512]],
      convs = [[(5, 32), (5, 64)], [(5, 64), (5, 128)]],
      dropout = [.65, .75, .85],
      decay_factor = [0.66, 0.5],
      decay_step = [100, 200]
    ),
    'mnist2': [
      Namespace(
        convs = [(5, 64), (5, 128)]
      )
    ],
    'crop': Namespace(
      lam = [0.0001, 0.0003, 0.001, 0.00003],
      alpha = [0.001, 0.003, 0.0001, 0.0003, 0.00003],
      fcs = [[1024], [512], [512, 512]],
      convs = [
        [(5, 32), (5, 64)],
        [(5, 64), (5, 128)],
        [(5, 32), (7, 64)],
        [(5, 16), (7, 512)],
        [(5, 32), (5, 64), (5, 64)]
      ],
      dropout = [.8, .85, .9],
      decay_step = [100, 80, 60]
    )
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
    variant='crop-small',
    train_data_name='crop-train-small',
    valid_data_name='crop-valid-small',
    test_data_name='crop-test-small',
    preprocessor='hog',
    param_set='crop'
  ),

  Namespace(
    model='baseline',
    variant='crop-big',
    train_data_name='crop-train-big',
    valid_data_name='crop-valid-big',
    test_data_name='crop-test-big',
    preprocessor='hog',
    param_set='crop'
  ),

  Namespace(
    model='baseline',
    variant='mnist',
    train_data_name='mnist-train',
    valid_data_name='mnist-valid',
    test_data_name='mnist-test',
    preprocessor='hog',
    param_set='mnist'
  ),

  Namespace(
    model='tf',
    variant='crop-huge',
    train_data_name='crop-train-huge',
    valid_data_name='crop-valid-huge',
    test_data_name='crop-test-big',
    preprocessor='color',
    param_set='crop'
  ),

  Namespace(
    model='tf',
    variant='crop-big',
    train_data_name='crop-train-big',
    valid_data_name='crop-valid-big',
    test_data_name='crop-test-big',
    preprocessor='color',
    param_set='crop'
  ),

  Namespace(
    model='tf',
    variant='crop-small',
    train_data_name='crop-train-small',
    valid_data_name='crop-valid-small',
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
    model='vote',
    variant='mnist',
    train_data_name='mnist-train',
    valid_data_name='mnist-valid',
    test_data_name='mnist-test',
    preprocessor='color',
    param_set='mnist'
  ),

  Namespace(
    model='vote',
    variant='crop-huge',
    train_data_name='crop-train-huge',
    valid_data_name='crop-valid-huge',
    test_data_name='crop-test-big',
    preprocessor='color',
    param_set='crop'
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
  ),

  Namespace(
    model='tf',
    variant='crop-big-search',
    train_data_name='crop-train-big',
    valid_data_name='crop-valid-big',
    test_data_name='crop-test-big',
    preprocessor='color',
    param_set='crop',
    search_set='crop',
    search_size=20
  ),

  Namespace(
    model='tf',
    variant='crop-huge-search',
    train_data_name='crop-train-huge',
    valid_data_name='crop-valid-huge',
    test_data_name='crop-test-big',
    preprocessor='color',
    param_set='crop',
    search_set='crop',
    search_size=20
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
