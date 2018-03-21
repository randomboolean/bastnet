"""Run grid search for a ChebNet architecture on Haxby data set"""

import os
import argparse
import shutil
import time
import numpy as np
from loader import load_graph
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from models import cgcnn
from graph import laplacian

parser = argparse.ArgumentParser(description="Run sim on cifar10 with ChebNet")
parser.add_argument('--graph', default='cifar10_cov_4closest_symmetrized', type=str)
parser.add_argument('--order', default=5, type=int)
args = parser.parse_args()

#
# Data
#

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train, X_test= X_train / 255.0 , X_test / 255.0
#y_train, y_test = to_categorical(y_train), to_categorical(y_test)
y_train, y_test = y_train.reshape((y_train.shape[0],)), ((y_test.shape[0],))

n_train = X_train.shape[0]
n_test= X_test.shape[0]
imgWidth = X_train.shape[1]
imgHeight = X_train.shape[2]
imgChannels = X_train.shape[3]
X_train, X_test= X_train.reshape(n_train, imgWidth*imgHeight, imgChannels) , X_test.reshape(n_test, imgWidth*imgHeight, imgChannels)
C = 10

#
# Graphs and laplacians
#

A = load_graph(args.graph)
graphs = [A]  # only one level and no pooling
L = [laplacian(A, normalized=True) for A in graphs]

#
# Spectral graph Cnn
#

params = dict()
params['dir_name'] = 'cifar10_K{}_{}'.format(args.order, args.graph) 
params['num_epochs'] = 150
params['batch_size'] = 32
params['eval_frequency'] = 100

# Building blocks.
params['filter'] = 'chebyshev5'
params['brelu'] = 'b1relu'
params['pool'] = 'mpool1'

# Number of classes.
assert C == 10

# Architecture.
params['F'] = [96,96,96,192,192,192,96]  # Number of graph convolutional filters. -> grid searched
params['D'] = [0.,0.,0.5,0.,0.5,0.5,0.5] # Dropout per conv filters
depth = len(params['F'])
params['K'] = [args.order] * depth  # Polynomial orders.
params['p'] = [1] * depth  # Pooling sizes.
params['M'] = [C]  # Output dimensionality of fully connected layers. -> grid searched
params['nb_channels'] = 3

# Optimization.
params['regularization'] = 5e-4
params['dropout'] = 0.
params['decay_rate'] = 0.95
params['momentum'] = 0.
params['decay_steps'] = n_train / params['batch_size']
params['verbose'] = False

#
# Run
#

filename = params['dir_name'] + '.log'
saver = open(filename, 'w')
saver.write("max time\n")

t_start = time.time()
model = cgcnn(L * depth, **params)
_accuracy, _loss, _t_step = model.fit(X_train, y_train, X_test, y_test, verbose=False)

maxValue = max(_accuracy)

# remove folder
shutil.rmtree('checkpoints/' + params['dir_name'])
shutil.rmtree('summaries/' + params['dir_name'])

t_spent = time.time() - t_start
saver.write("{} {:.2f}s\n".format(maxValue, t_spent))
saver.flush()
