def fildir(x):
    return filter(lambda n: not n.startswith('__'), dir(x))
