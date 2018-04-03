#!/usr/bin/env python3.5

import os
import numpy as np
from datetime import datetime
from LumpyBgnd import lumpy_backround, create_lumps_pos_matrix


def get_current_time(time=True, date=False):
    now = datetime.now()
    s = ""
    if date:
        s += "{} ".format(now.date())
    if time:
        s += "{:02d}:{:02d}:{:02d}".format(now.hour, now.minute, now.second)
    return s.strip()


def generate_lumps_matrix(train_size=60000, test_size=10000, verbose=False):
    dim = (64, 64)
    nbar = 200
    dc = 10
    lump_function = "GaussLmp"
    pars = (1, 2)
    discrete_lumps = False
    range_values = (0, 1)

    fdbk_percentage = 2
    if verbose:
        print_feedback = int(train_size * fdbk_percentage / 100)
    else:
        print_feedback = train_size + 1
    x_train = np.empty((train_size, dim[0], dim[1]))
    y_train = np.empty((train_size, dim[0], dim[1]))
    for i in range(train_size):
        image, n, lumps_pos = lumpy_backround(dim=dim, nbar=nbar, dc=dc, pars=pars,
                                              lump_function=lump_function, rng=range_values,
                                              discretize_lumps_positions=discrete_lumps)
        pos_matrix = create_lumps_pos_matrix(dim=dim, lumps_pos=lumps_pos)
        x_train[i] = pos_matrix
        y_train[i] = image
        if (i + 1) % print_feedback == 0:
            print("{}% of training dataset generated ({} samples)".format(
                (i + 1) * 100 / train_size, i + 1))
    if verbose:
        print_feedback = int(test_size * fdbk_percentage / 100)
    else:
        print_feedback = test_size + 1
    x_test = np.empty((test_size, dim[0], dim[1]))
    y_test = np.empty((test_size, dim[0], dim[1]))
    for i in range(test_size):
        image, n, lumps_pos = lumpy_backround(dim=dim, nbar=nbar, dc=dc, pars=pars,
                                              lump_function=lump_function, rng=range_values,
                                              discretize_lumps_positions=discrete_lumps)
        pos_matrix = create_lumps_pos_matrix(dim=dim, lumps_pos=lumps_pos)
        x_test[i] = pos_matrix
        y_test[i] = image
        if (i + 1) % print_feedback == 0:
            print("{}% of test dataset generated ({} samples)".format((i + 1) * 100 / test_size,
                                                                      i + 1))
    return (x_train, y_train), (x_test, y_test)


def load_data(dataset, filename_out=None):
    if filename_out is None:
        filename_out = dataset
    if dataset == "lumps":
        path = "./dataset/"
    elif dataset == "lumps_matrix":
        data = generate_lumps_matrix(train_size=60000, test_size=10000, verbose=True)
        np.save(filename_out, data)
        return
    else:
        raise KeyError("Unknown dataset: {}".format(dataset))
    files = sorted(os.listdir(path))
    num_files = len(files)
    percent = 2
    split_distance = num_files * percent // 100
    print("Loading {} samples from dataset {}".format(num_files, dataset))
    data = []
    for i, filename in enumerate(files):
        with open(path + filename, "r") as f:
            content = f.read()
            sample = np.array([[float(n) for n in line.split(",")] for line in content.split("\n") if len(line) > 0])
            data.append(sample)
        if (i + 1) % split_distance == 0:
            print("{}. {}% loaded".format(get_current_time(), (i + 1) * 100 // num_files))
    data = np.array(data)
    print("Shape:", data.shape)
    np.save(filename_out, data)


if __name__ == "__main__":
    # load_data("lumps")
    load_data("lumps_matrix")
