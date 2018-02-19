"""Run grid search for a ChebNet architecture on Haxby data set"""

import os
import argparse
import shutil
import time
import numpy as np
from loader import load_dataset, load_graph
from models import cgcnn
from graph import laplacian

parser = argparse.ArgumentParser(description="Run grid search for ChebNet on Haxby")
parser.add_argument('--graph', default='haxby_geo_6closest_symmetrized', type=str)
parser.add_argument('--subject', default=0, type=int)
parser.add_argument('--order', default=20, type=int)
args = parser.parse_args()

#
# Data
#

(X_train, y_train), (X_test, y_test) = load_dataset(subject=args.subject)
X_val, y_val = X_test, y_test
n_train, d = X_train.shape
n_val = y_val.shape
C = y_val.max() + 1

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
params['dir_name'] = None  # will be changed during grid searched
params['num_epochs'] = 100
params['batch_size'] = 32
params['eval_frequency'] = 100

# Building blocks.
params['filter'] = 'chebyshev5'
params['brelu'] = 'b1relu'
params['pool'] = 'mpool1'

# Number of classes.
assert C == 8

# Architecture.
params['F'] = None  # Number of graph convolutional filters. -> grid searched
params['K'] = [args.order]  # Polynomial orders.
params['p'] = [1]  # Pooling sizes.
params['M'] = None  # Output dimensionality of fully connected layers. -> grid searched

# Optimization.
params['regularization'] = 5e-4
params['dropout'] = None  # -> grid searched
params['learning_rate'] = 1e-3
params['decay_rate'] = 0.95
params['momentum'] = 0.
params['decay_steps'] = n_train / params['batch_size']
params['verbose'] = False

#
# Grid Search
#

filename = 'subject{}_K{}_{}.log'.format(args.subject, args.order, args.graph)
if os.path.isfile(filename):
    last = open(filename, 'r').readlines()[-1].split(' ')
    resumer = [int(last[0]), int(last[1]), int(last[2]), float(last[3])]
    gridSearchMax = float(last[4])
    gridSearchAveStd = float(last[5]), float(last[6])
    saver = open(filename, 'a')
else:
    resumer = None
    saver = open(filename, 'w')
    saver.write("F M1 M2 dropout cur_max cur_ave cur_std time\n")
    gridSearchMax = 0.
    gridSearchAveStd = 0., 0.
sim_id = -1
NB_INITIALIZATIONS = 25
resume = resumer is None
for F in [8, 16, 32, 64, 128]:
    for M1 in [32, 64, 128]:
        for M2 in [0, 32, 64, 128]:
            for dropout in [1., 0.75, 0.5]:

                # Do we resume from last checkpoint or not
                if not(resume):
                    if [F, M1, M2, dropout] == resumer:
                        resume = True
                    sim_id += NB_INITIALIZATIONS
                    continue

                # We keep the best max and best ave found
                results = []
                t_start = time.time()
                for init in range(NB_INITIALIZATIONS):
                    sim_id += 1
                    params['dir_name'] = 'subject{}_K{}_{}_{}'.format(args.subject, args.order, args.graph, sim_id) 
                    params['F'] = [F]
                    params['M'] = [M1, M2, C] if M2 else [M1, C]
                    params['dropout'] = dropout
                    model = cgcnn(L, **params)
                    _accuracy, _loss, _t_step = model.fit(X_train, y_train, X_val, y_val, verbose=False)

                    maxValue = max(_accuracy)
                    results += [maxValue]
                    if maxValue > gridSearchMax:
                        gridSearchMax = maxValue
                        accuracy, loss, t_step = _accuracy, _loss, _t_step
                        toprint = "Current best Max: " + str(gridSearchMax) + " (cnnLayerNbFeatureMaps=" + str(
                            F) + ", fullyConnectedSize1=" + str(M1) + ", fullyConnectedSize2=" + str(
                            M2) + ", dropout=" + str(dropout) + ")"
                        print(toprint)

                    # remove folder
                    shutil.rmtree('checkpoints/' + params['dir_name'])
                    shutil.rmtree('summaries/' + params['dir_name'])
                aveStd = np.mean(results), np.std(results)
                if aveStd[0] > gridSearchAveStd[0]:
                    gridSearchAveStd = aveStd
                    toprint = "Current best AverageStd: " + str(gridSearchAveStd) + " (cnnLayerNbFeatureMaps=" + str(
                        F) + ", fullyConnectedSize1=" + str(M1) + ", fullyConnectedSize2=" + str(
                        M2) + ", dropout=" + str(dropout) + ")"
                    print(toprint)

                t_spent = time.time() - t_start
                saver.write("{} {} {} {} {} {} {} {:.2f}s\n".format(F, M1, M2, dropout, gridSearchMax, gridSearchAveStd[0], gridSearchAveStd[1], t_spent))
                saver.flush()

saver.write("Best Max found: " + str(gridSearchMax) + '\n')
saver.write("Best AveStd found: " + str(gridSearchAveStd) + '\n')
saver.flush()
