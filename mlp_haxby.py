"""Run grid search for a ChebNet architecture on Haxby data set"""

import os
import argparse
import shutil
import time
import numpy as np
from loader import load_dataset, load_graph
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

parser = argparse.ArgumentParser(description="Run grid search for MLP on Haxby")
parser.add_argument('--subject', '-s', default=0, type=int)
args = parser.parse_args()

(X_train, y_train), (X_test, y_test) = load_dataset(subject=args.subject)
X_val, y_val = X_test, y_test
n_train, d = X_train.shape
C = 8

filename = 'MLP_subject{}.log'.format(args.subject)
if os.path.isfile(filename):
    last = open(filename, 'r').readlines()[-1].split(' ')
    resumer = [int(last[0]), int(last[1]), float(last[2])]
    gridSearchMax = float(last[3])
    gridSearchAveStd = float(last[4]), float(last[5])
    saver = open(filename, 'a')
else:
    resumer = None
    saver = open(filename, 'w')
    saver.write("M1 M2 dropout cur_max cur_ave cur_std time\n")
    gridSearchMax = 0.
    gridSearchAveStd = 0., 0.
sim_id = -1
NB_INITIALIZATIONS = 25
resume = resumer is None
for unused in [1]:
    for M1 in [128, 256, 512, 1024]:
        for M2 in [0, 128, 256, 512, 1024]:
            for dropout in [1., 0.8, 0.6]:

                # Do we resume from last checkpoint or not
                if not(resume):
                    if [M1, M2, dropout] == resumer:
                        resume = True
                    sim_id += NB_INITIALIZATIONS
                    continue

                # We keep the best max and best ave found
                results = []
                t_start = time.time()
                for init in range(NB_INITIALIZATIONS):
                    sim_id += 1
                    
                    model = Sequential()
                    model.add(Dense(M1, activation="relu", input_dim=d))
                    model.add(Dropout(dropout))
                    if M2 > 0:
                        model.add(Dense(M2, activation="relu"))
                        model.add(Dropout(dropout))
                    model.add(Dense(C, activation="softmax"))

                    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001, decay=1e-6), metrics=["accuracy"])
                    model.summary()
                    history = model.fit(X_train, to_categorical(y_train), batch_size=32, shuffle=True, epochs=50, validation_data=(X_test, to_categorical(y_test)), verbose=0)
                    
                    maxValue = max(history.history["val_acc"])
                    results += [maxValue]
                    if maxValue > gridSearchMax:
                        gridSearchMax = maxValue
                        toprint = "Current best Max: " + str(gridSearchMax) + " (fullyConnectedSize1=" + str(M1) + ", fullyConnectedSize2=" + str(
                            M2) + ", dropout=" + str(dropout) + ")"
                        print(toprint)

                aveStd = np.mean(results), np.std(results)
                if aveStd[0] > gridSearchAveStd[0]:
                    gridSearchAveStd = aveStd
                    toprint = "Current best AverageStd: " + str(gridSearchAveStd) + " (fullyConnectedSize1=" + str(M1) + ", fullyConnectedSize2=" + str(
                        M2) + ", dropout=" + str(dropout) + ")"
                    print(toprint)

                t_spent = time.time() - t_start
                saver.write("{} {} {} {} {} {} {:.2f}s\n".format(M1, M2, dropout, gridSearchMax, gridSearchAveStd[0], gridSearchAveStd[1], t_spent))
                saver.flush()

saver.write("Best Max found: " + str(gridSearchMax) + '\n')
saver.write("Best AveStd found: " + str(gridSearchAveStd) + '\n')
saver.flush()
