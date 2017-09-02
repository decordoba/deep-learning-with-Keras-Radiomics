#!/usr/bin/env python3.5

import os
import yaml
import itertools
from time import clock
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from keras_utils import format_dataset  # 'Library' by Daniel


class Experiment(ABC):
    """
    Abstract class to set the parameters that we want to run
    """
    @abstractmethod
    def __init__(self):
        """
        Expects self.experiments to be defined
        Example:
            self.experiments = {"filters1": [16, 32, 64],
                                "filters2": [16, 32, 64],
                                "losses1": [losses.MSE, losses.MAE, losses.hinge, losses.categorical_crossentropy],
                                "units1": [16, 32, 64],
                                "optimizers1": [optimizers.Adam()],
                                "kernel_sizes1": [(3, 3)],
                                "dropouts1": [0.25],
                                "dropouts2": [0.5],
                                "pool_sizes1": [(2, 2)]}
        """
        self.experiments = None

    @abstractmethod
    def run_experiment(self, *args):
        """
        Expects self.experiments to be used to run all the experiments. Designed to easily work
        with keras_utils.flexible_neural_net, although it should be flexible to work with any
        other function.
        Example:
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
        """
        pass

    def get_experiments(self):
        # count the number of iterations to do all experiments
        self.num_iterations = 1
        for key in self.experiments:
            if not isinstance(self.experiments[key], list):
                self.experiments[key] = [self.experiments[key]]
            self.num_iterations *= len(self.experiments[key])
        # generate structures to create iterable object
        self.experiments_keys = sorted(self.experiments.keys())
        self.keys_mapper = dict([(key, i) for i, key in enumerate(self.experiments_keys)])
        self.comb = [self.experiments[key] for key in self.experiments_keys]
        return itertools.product(*self.comb), self.num_iterations

    def get_printable_experiment(self, comb, iter_num, verbose=False):
        # make sure that self.get_experiments() has been executed before
        try:
            self.experiments_keys
        except AttributeError:
            self.get_experiments()

        params = {}
        # convert the functions and objects into something printable
        for key, val in zip(self.experiments_keys, comb):
            if callable(val):
                try:
                    val = str(val).split()[1]
                except:
                    val = str(val)
            elif not isinstance(val, (int, float, str, tuple, list, dict, set)):
                if str(val).startswith("<"):
                    # val = val.__class__.__name__  # Returns class name (i.e. Adam)
                    val = str(type(val))
                    if val.startswith("<class '"):
                        val = val.split("'")[1]     # Returns whole class name (i.e. keras.optimizers.Adam)
                    # try:
                    #     opt_txt = str(opt).split(".")[2].split()[0]
                    # except:
                    #     opt_txt = str(opt)
                else:
                    val = str(val)
            params[key] = val

        # add informative parameters not related with the combination
        params["iteration_number"] = iter_num
        now = datetime.now()
        params["date"] = "{} {:02d}:{:02d}:{:02d}".format(now.date(), now.hour, now.minute, now.second)
        # params["location"] = location

        if verbose:
            for key in self.experiments_keys:
                print("  {:<15}{}".format(key + ":", params[key]))
            print("  {:<15}{}".format("date" + ":", params["date"]))

        return params


def params_in_data(params, data, ignore_keys=("iteration_number", "date", "location")):
    """
    Returns if params are found in data (if so it will return the location) or False otherwise
    This is not the most optimal code, I know, this function will be called many times and
    slow operations like deleting elements in an unsorted list for every sample will be repeated
    many times. It should not matter because keys_params should not have more than 20 keys, so I
    think it is not worth it to create a data structure to avoid repeating the same operation
    """
    if data is None:
        return False
    keys_params = list(params.keys())
    for key in ignore_keys:
        try:
            keys_params.remove(key)
        except ValueError:
            pass
    # search for experiment in data
    for loc in data:  # Expects a specific structure in data: a dict with location as a key
        sample = data[loc]["params"]
        keys_sample = list(sample.keys())
        for key in ignore_keys:
            try:
                keys_sample.remove(key)
            except ValueError:
                pass
        if len(keys_sample) != len(keys_params):
            print("Data has a different number of parameters than Params. Cannot be compared...")
            print("Data params ({}):   {}\nParams params ({}): {}".format(len(keys_sample),
                                                                          keys_sample,
                                                                          len(keys_params),
                                                                          keys_params))
            continue
        else:
            same = True
            for key in keys_params:
                try:
                    if sample[key] != params[key]:
                        same = False
                        break
                except KeyError:
                    print("Data has different parameters than Params. Cannot be compared...")
                    print("Data params ({}):   {}\nParams params ({}): {}".format(len(keys_sample),
                                                                                  keys_sample,
                                                                                  len(keys_params),
                                                                                  keys_params))
                    same = False
                    break
            if same:
                return loc
    return False

def experiments_runner(data_generator, experiment_obj, folder=None, data_reduction=None,
                                epochs=100, to_categorical=True):
    print("Loading training and test sets ...")
    # Load into train and test sets
    (x_train, y_train), (x_test, y_test) = data_generator()

    print("Reshaping training and test sets ...")
    train_set, test_set, input_shape, labels = format_dataset(x_train, y_train, x_test, y_test,
                                                              data_reduction=data_reduction,
                                                              verbose=True, ret_labels=True,
                                                              to_categorical=to_categorical)

    # create folder and cd into it
    if folder is None:
        now = datetime.now()
        folder = "{}_{:02d}.{:02d}.{:02d}".format(now.date(), now.hour, now.minute, now.second)
    try:
        os.makedirs(folder)
    except OSError:
        pass    # In case the dir already exists
    os.chdir(folder)

    # load data in old results.yaml, if it exists
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

    # train model and save data
    print("Generating and training models...")
    experiment = experiment_obj()
    iterator, num_iterations = experiment.get_experiments()
    avg_time = 0
    num_skips = 0
    for it, params_comb in enumerate(iterator):
        t = clock()
        print("\niteration: {}/{}".format(it + 1, num_iterations))
        params = experiment.get_printable_experiment(params_comb, it + 1, verbose=True)

        # skip experiments that are already found in old results.yaml
        skip_test = params_in_data(params, old_data)
        if skip_test != False:
            print("The folder {} already contains this model (found in {}). Model calculation skipped.".format(folder, skip_test))
            num_skips += 1
            continue

        # run experiments (returns same as flexible_neural_network)
        [lTr, aTr], [lTe, aTe], time, location, n_epochs = experiment.run_experiment(train_set, test_set,
                                                                                     input_shape, labels,
                                                                                     params_comb, epochs)
        result = {"lossTr": float(lTr), "accTr": float(aTr),
                  "lossTe": float(lTe), "accTe": float(aTe),
                  "time": float(time), "location": location,
                  "number_epochs": n_epochs}

        # save results to result.yaml
        with open("results.yaml", "a") as f:
            f.write(yaml.dump_all([{location: {"params": params,
                                               "result": result}}],
                                  default_flow_style=False,
                                  explicit_start=False))

        # print data to monitor how well we are doing
        taken = clock() - t
        avg_time = (avg_time * (it - num_skips) + taken) / (it - num_skips + 1)
        print("\nResults:  Training:  Acc: {:10}  Loss: {}".format(aTr, lTr))
        print("          Test:      Acc: {:<10}  Loss: {}".format(aTe, lTe))
        print("          Time taken:         {}  (fit & evaluation time: {})".format(timedelta(seconds=taken),
                                                                                     timedelta(seconds=time)))
        print("          Expected time left: {}  (mean time: {})".format(timedelta(seconds=avg_time * (num_iterations - it - 1)),
                                                                         timedelta(seconds=avg_time)))

    os.chdir("./..")
    return folder
