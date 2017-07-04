#!/usr/bin/env python3.5

import sys
from time import clock
from datetime import timedelta
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import losses
from keras import optimizers
from keras.datasets import mnist
from keras_utils import flexible_neural_net  # 'Library' by Daniel
from keras_experiments import Experiment, experiments_runner  # 'Library' by Daniel


class MyFirstExperiment(Experiment):
    def __init__(self):
        # filters1 = [4, 8, 16, 32, 64, 128, 256]
        # filters2 = [4, 8, 16, 32, 64, 128, 256]
        # optimizers1 = [optimizers.Adadelta(), optimizers.Adagrad(), optimizers.Adam(), optimizers.Adamax(), optimizers.SGD(), optimizers.RMSprop()]
        # losses1 = [losses.MSE, losses.MAE, losses.hinge, losses.categorical_crossentropy]
        # units1 = [4, 8, 16, 32, 64, 128, 256]
        # kernel_sizes = [(3, 3), (5, 5)]
        # dropouts1 = [0.25, 0.5, 0.75]
        # dropouts2 = [0.25, 0.5, 0.75]
        # pool_sizes1 = [(2, 2)]

        filters1 = [16, 32, 64]
        filters2 = [16, 32, 64]
        losses1 = [losses.MSE, losses.MAE, losses.hinge, losses.categorical_crossentropy]
        units1 = [16, 32, 64]
        optimizers1 = [optimizers.Adam()]
        kernel_sizes1 = [(3, 3)]
        dropouts1 = [0.25]
        dropouts2 = [0.5]
        pool_sizes1 = [(2, 2)]

        # create standard experiments structure
        self.experiments = {"filters1": filters1,
                            "filters2": filters2,
                            "losses1": losses1,
                            "units1": units1,
                            "optimizers1": optimizers1,
                            "kernel_sizes1": kernel_sizes1,
                            "dropouts1": dropouts1,
                            "dropouts2": dropouts2,
                            "pool_sizes1": pool_sizes1}

    def run_experiment(self, train_set, test_set, input_shape, labels, comb, epochs):
        try:
            self.keys_mapper
        except AttributeError:
            self.get_experiments()
        opt = comb[self.keys_mapper["optimizers1"]]
        loss = comb[self.keys_mapper["losses1"]]
        f1 = comb[self.keys_mapper["filters1"]]
        f2 = comb[self.keys_mapper["filters2"]]
        u1 = comb[self.keys_mapper["units1"]]
        ks = comb[self.keys_mapper["kernel_sizes1"]]
        ps = comb[self.keys_mapper["pool_sizes1"]]
        d1 = comb[self.keys_mapper["dropouts1"]]
        d2 = comb[self.keys_mapper["dropouts2"]]
        return flexible_neural_net(train_set, test_set, opt, loss,
                                   Conv2D(f1, kernel_size=ks, activation='relu', input_shape=input_shape),
                                   Conv2D(f2, kernel_size=ks, activation='relu'),
                                   MaxPooling2D(pool_size=ps),
                                   Dropout(d1),
                                   Flatten(),
                                   Dense(u1, activation='relu'),
                                   Dropout(d2),
                                   Dense(len(labels), activation='softmax'),
                                   batch_size=32, epochs=epochs,
                                   verbose=False)


class MyFirstExperimentContinued(Experiment):
    def __init__(self):
        # create standard experiments structure
        self.experiments = {"filters1": [16, 32, 64, 128, 256],
                            "filters2": [16, 32, 64, 128, 256],
                            "losses1": [losses.categorical_crossentropy],
                            "units1": [16, 32, 64, 128, 256],
                            "optimizers1": [optimizers.Adam()],
                            "kernel_sizes1": [(3, 3)],
                            "dropouts1": [0.25],
                            "dropouts2": [0.5],
                            "pool_sizes1": [(2, 2)]}

    def run_experiment(self, train_set, test_set, input_shape, labels, comb, epochs):
        opt = comb[self.keys_mapper["optimizers1"]]
        loss = comb[self.keys_mapper["losses1"]]
        f1 = comb[self.keys_mapper["filters1"]]
        f2 = comb[self.keys_mapper["filters2"]]
        u1 = comb[self.keys_mapper["units1"]]
        ks = comb[self.keys_mapper["kernel_sizes1"]]
        ps = comb[self.keys_mapper["pool_sizes1"]]
        d1 = comb[self.keys_mapper["dropouts1"]]
        d2 = comb[self.keys_mapper["dropouts2"]]
        return flexible_neural_net(train_set, test_set, opt, loss,
                                   Conv2D(f1, kernel_size=ks, activation='relu', input_shape=input_shape),
                                   Conv2D(f2, kernel_size=ks, activation='relu'),
                                   MaxPooling2D(pool_size=ps),
                                   Dropout(d1),
                                   Flatten(),
                                   Dense(u1, activation='relu'),
                                   Dropout(d2),
                                   Dense(len(labels), activation='softmax'),
                                   batch_size=32, epochs=epochs,
                                   verbose=False)


if __name__ == "__main__":

    t = clock()

    experiment = MyFirstExperimentContinued
    data = mnist.load_data

    if len(sys.argv) == 1:
        experiments_runner(data, experiment)
    elif len(sys.argv) == 2:
        experiments_runner(data, experiment, folder=sys.argv[1])
    elif len(sys.argv) == 3:
        experiments_runner(data, experiment, folder=sys.argv[1], data_reduction=int(sys.argv[2]))
    elif len(sys.argv) >= 4:
        experiments_runner(data, experiment, folder=sys.argv[1], data_reduction=int(sys.argv[2]),
                           epochs=int(sys.argv[3]))

    print("\nTotal Time Taken: {} s".format(timedelta(seconds=clock() - t)))
