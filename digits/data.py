from collections import namedtuple
import os
import pickle

import numpy as np
import scipy.io as spio

Data = namedtuple('Data', ['dataset', 'role', 'X', 'y'])

class Loader:
    def __init__(self, path):
        self.path = path
        self.data_prefix = os.path.join(self.path,'data')
        self.pickle_prefix = os.path.join(self.path,'pickled')
        self.mat_suffix = '_32x32.mat'
        self.pickle_suffix = '.pickle'
        
    def assert_ready(self):
        assert os.path.isdir(self.data_prefix)
        assert os.path.isdir(self.pickle_prefix)
        
    def clear_pickled(self):
        files = os.listdir(self.pickle_prefix)
        for file in files:
            if file.endswith('.pickle'):
                os.remove(os.path.join(self.pickle_prefix, file))
        
    # TODO support more than cropped
    def read_cropped(self, role):
        mat_file = os.path.join(self.data_prefix, role + self.mat_suffix)
        pickle_file = os.path.join(self.pickle_prefix, role + self.mat_suffix + self.pickle_suffix)
        if not os.path.isfile(pickle_file):
            mat = spio.loadmat(mat_file)
            X = mat['X']
            y = mat['y']
            assert len(mat['X'].shape) == 4
            assert len(mat['y'].shape) == 2
            assert mat['X'].shape[-1] == mat['y'].shape[0]
            assert mat['y'].shape[1] == 1
            # Cleanup X: make row-oriented :(
            X = np.moveaxis(X, 3, 0)
            assert X.shape[0] == y.shape[0]
            # Cleanup y: '10' really means '0' :(
            for i in range(len(y)):
              if y[i][0] == 10:
                y[i][0] = 0
            data = Data(dataset='cropped', role=role, X=X, y=y)
            with open(pickle_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            return data
        else:
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)
          
    def raw_pickle_file(self, name):
        return os.path.join(self.pickle_prefix, name + self.pickle_suffix)
        
    def write_raw(self, name, raw):
        with open(self.raw_pickle_file(name), 'wb') as f:
            pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    def raw_exists(self, name):
        return os.path.isfile(self.raw_pickle_file(name))
    
    def del_raw(self, name):
        os.remove(self.raw_pickle_file(name))
        
    def read_raw(self, name):
        with open(self.raw_pickle_file(name), 'rb') as f:
            return pickle.load(f)
