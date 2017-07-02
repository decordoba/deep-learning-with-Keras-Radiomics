#!/usr/bin/env python3.5

import sys
import os
import yaml
import itertools
from time import clock
from datetime import datetime, timedelta
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import losses
from keras import optimizers
from keras.datasets import mnist
from keras_utils import format_dataset, flexible_neural_net  # 'Library' by Daniel


class Experiment0():
    def params_in_data(self, params, data):
        if data is None:
            return False
        for loc in data:
            sample = data[loc]
            if (sample["optimizer"] == params["optimizer"] and
                    sample["loss"] == params["loss"] and
                    sample["filter1"] == params["filter1"] and
                    sample["filter2"] == params["filter2"] and
                    sample["unit1"] == params["unit1"] and
                    sample["kernel_size"] == params["kernel_size"] and
                    sample["pool_size"] == params["pool_size"] and
                    sample["dropout_rate1"] == params["dropout_rate1"] and
                    sample["dropout_rate2"] == params["dropout_rate2"]):
                return loc
        return False

    def get_experiments(self):
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
        experiments = {"filters1": filters1,
                       "filters2": filters2,
                       "losses1": losses1,
                       "units1": units1,
                       "optimizers1": optimizers1,
                       "kernel_sizes1": kernel_sizes1,
                       "dropouts1": dropouts1,
                       "dropouts2": dropouts2,
                       "pool_sizes1": pool_sizes1}

        # count the number of iterations
        num_iterations = 1
        for key in experiments:
            num_iterations *= len(experiments[key])

        # generate structures to create iterable object
        experiments_keys = sorted(experiments.keys())
        comb = [experiments[key] for key in experiments_keys]

        return experiments, experiments_keys, itertools.product(*comb), num_iterations

    def get_params_experiment(self, comb, keys, iter_num, verbose=False):
        params = {}
        for key, val in zip(keys, comb):
            if callable(val):
                try:
                    val = str(val).split()[1]
                except:
                    val = str(val)
            elif isinstance(val, (int, float, str, tuple, list, dict, set)):
                pass
            else:
                if str(val)[0] == "<":
                    val = type(val)
                    # try:
                    #     opt_txt = str(opt).split(".")[2].split()[0]
                    # except:
                    #     opt_txt = str(opt)
                else:
                    val = str(val)
            params[key] = val

        # add informative parameters not related with the combination
        # params["location"] = location
        params["iteration_number"] = iter_num
        now = datetime.now()
        params["date"] = "{} {:02d}:{:02d}:{:02d}".format(now.date(), now.hour, now.minute, now.second)

        if verbose:
            for key in keys:
                print("{:<15}{}".format(key + ":", params[keys]))

        return params


def main(folder=None, data_reduction=None, epochs=100):
    print("Loading training and test sets ...")
    # Load pre-shuffled MNIST data into train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Reshaping training and test sets ...")
    train_set, test_set, input_shape, labels = format_dataset(x_train, y_train, x_test, y_test,
                                                              data_reduction=data_reduction,
                                                              verbose=True, to_categorical=True,
                                                              ret_labels=True)

    if folder is None:
        now = datetime.now()
        folder = "{}_{:02d}.{:02d}.{:02d}".format(now.date(), now.hour, now.minute, now.second)
    try:
        os.makedirs(folder)
    except OSError:
        pass    # In case the dir already exists
    os.chdir(folder)

    try:
        with open("results.yaml") as f:
            try:
                old_data = yaml.load(f)
                print("'results.yaml' was parsed successfully. The experiments that appear in " +
                      "'results.yaml' will not be executed again.")
            except yaml.YAMLError as YamlError:
                print("There was an error parsing 'results.yaml'. File ignored.")
                print(YamlError)
                old_data = None
    except FileNotFoundError:
        old_data = None

    print("Generating and training models...")
    experiments, experiment_keys, iterator, num_iterations = get_experiments()

    avg_time = 0
    for it, params_comb in enumerate(iterator):
        t = clock()
        print("\nIteration {}/{}".format(it + 1, num_iterations))
        params = get_params_experiment(params_comb, experiment_keys, it + 1, verbose=True)

        # Skip experiments that are already found in old results.yaml
        skip_test = params_in_data(params, old_data)
        if skip_test != False:
            print("The folder {} already contains this model (found in {}). Model calculation skipped".format(location, skip_test))
            continue

        [lTr, aTr], [lTe, aTe], time, location = flexible_neural_net(train_set, test_set, opt, loss,
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
        result = {"lossTr": float(lTr), "accTr": float(aTr),
                  "lossTe": float(lTe), "accTe": float(aTe),
                  "time": float(time), "location": location}

        with open("results.yaml", "a") as f:
            f.write(yaml.dump_all([{location: {"params": params,
                                               "result": result}}],
                                  default_flow_style=False,
                                  explicit_start=False))

        taken = clock() - t
        avg_time += (taken / it)
        print("\nResults:  Training:  Loss: {}  Acc: {}".format(lTr, aTr))
        print("          Test:      Loss: {}  Acc: {}".format(lTe, aTe))
        print("          Time taken: {}  (fit & evaluation time: {})".format(timedelta(seconds=taken), timedelta(seconds=time)))
        print("          Expected time left: {}".format(timedelta(seconds=avg_time * (num_iterations - it))))


    # flexible_neural_net(train_set, test_set, optimizers.Adam(), losses.MSE,
    #                     Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    #                     Conv2D(64, kernel_size=(3, 3), activation='relu'),
    #                     MaxPooling2D(pool_size=(2, 2)),
    #                     Dropout(0.25),
    #                     Flatten(),
    #                     Dense(128, activation='relu'),
    #                     Dropout(0.5),
    #                     Dense(len(labels), activation='softmax'))

    os.chdir("./..")


if __name__ == "__main__":

    t = clock()

    if len(sys.argv) == 1:
        main()
    elif len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], int(sys.argv[2]))
    elif len(sys.argv) == 4:
        main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    else:
        main(int(sys.argv[1]), int(sys.argv[2]), True)

    print("\nTotal Time Taken: {} s".format(timedelta(seconds=clock() - t)))
