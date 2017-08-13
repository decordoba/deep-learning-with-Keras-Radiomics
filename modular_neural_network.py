#!/usr/bin/env python3.5

import sys
from time import clock
from datetime import timedelta
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import losses
from keras import optimizers
from keras.datasets import mnist, cifar10
from keras_utils import flexible_neural_net  # 'Library' by Daniel
from keras_experiments import Experiment, experiments_runner  # 'Library' by Daniel


class MyFirstExperiment(Experiment):
    def __init__(self):
        filters1 = [16, 32, 64]  # filters1 = [4, 8, 16, 32, 64, 128, 256]
        filters2 = [16, 32, 64]  # filters2 = [4, 8, 16, 32, 64, 128, 256]
        losses1 = [losses.MSE, losses.MAE, losses.hinge, losses.categorical_crossentropy]  # losses1 = [losses.MSE, losses.MAE, losses.hinge, losses.categorical_crossentropy]
        optimizers1 = [optimizers.Adam()]  # optimizers1 = [optimizers.Adadelta(), optimizers.Adagrad(), optimizers.Adam(), optimizers.Adamax(), optimizers.SGD(), optimizers.RMSprop()]
        units1 = [16, 32, 64]  # units1 = [4, 8, 16, 32, 64, 128, 256]
        kernel_sizes1 = [(3, 3)]  # kernel_sizes = [(3, 3), (5, 5)]
        dropouts1 = [0.25]  # dropouts1 = [0.25, 0.5, 0.75]
        dropouts2 = [0.5]  # dropouts2 = [0.25, 0.5, 0.75]
        pool_sizes1 = [(2, 2)]  # pool_sizes1 = [(2, 2)]

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
        # comb holds values like (32, (2,2), optimizers-Adam()). We need to use self.keys_mapper
        # which maps a name ("units", "kernel_sizes", "optimizers") to the position where it is
        # in comb. I wonder if it would be more comprehensible with a function like
        # get_element_from_comb(self, comb, key) { return comb[self.keys_mapper[key]] }
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


class MyFirstExperimentContinued(MyFirstExperiment):
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


class MyFirstExperimentShort(MyFirstExperiment):
    def __init__(self):
        # create standard experiments structure
        self.experiments = {"filters1": [16, 32, 64],
                            "filters2": [16, 32, 64],
                            "losses1": [losses.categorical_crossentropy],
                            "units1": [16, 32],
                            "optimizers1": [optimizers.Adam()],
                            "kernel_sizes1": [(3, 3)],
                            "dropouts1": [0.25],
                            "dropouts2": [0.5],
                            "pool_sizes1": [(2, 2)]}


class SingleExperiment(MyFirstExperiment):
    def __init__(self):
        # create standard experiments structure
        self.experiments = {"filters1": 32,
                            "filters2": 32,
                            "losses1": losses.categorical_crossentropy,
                            "units1": 32,
                            "optimizers1": optimizers.Adam(),
                            "kernel_sizes1": (3, 3),
                            "dropouts1": 0.25,
                            "dropouts2": 0.5,
                            "pool_sizes1": (2, 2)}


if __name__ == "__main__":

    t = clock()

    experiment = SingleExperiment
    # data = mnist.load_data
    data = cifar10.load_data

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

    """
    Expects:
        py modular_neural_network.py
        py modular_neural_network.py folder
        py modular_neural_network.py folder data_reduction
        py modular_neural_network.py folder data_reduction epochs
    """