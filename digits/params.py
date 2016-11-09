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
      lam =  0.0001,
      # learning rate
      alpha = 0.001,
      # TODO 200k for mnist
      training_iters = 200000,  
      # number of examples per descent
      batch_size = 128,
      # number of examples per display step
      display_size = 512,
      # number of batches per display/validation step
      display_step = 10,
      # keep_prob, 1.0 keep all
      dropout = 0.75,
      # (width, depth) of convolutional layers
      convs = [(5, 32), (5, 64)],
      # size of fully connected layers
      fcs = [1024],
      # randomize image rotation, etc
      use_rando = False
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
    )
  }
}
