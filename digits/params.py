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
      lam =  0.00000001,
      # learning rate
      alpha = 0.001,
      # TODO 200k for mnist
      training_iters = 10000,  
      # number of examples per descent
      batch_size = 128,
      # number of batches per display/validation step
      display_step = 10,
      # keep_prob, 1.0 keep all
      dropout = 0.75,
      # (width, depth) of convolutional layers
      convs = [(5, 32), (5, 64)],
      # size of fully connected layers
      fcs = [1024]
    ),
    'crop': Namespace(
      num_classes = 10
    )
  }
}
