#!/usr/bin/env python3.5

import sys
import os
import yaml
import numpy as np
from dataset_loader import load_dataset, load_patients_dataset
from keras.models import model_from_yaml
from keras_utils import format_dataset  # 'Library' by Daniel

"""
This code adds vol_accTr and vol_AccTe to every experiment in results, so they can be shown with
results_plotter
"""


def load_model(location):
    with open(location + "/model.yaml", "r") as f:
        model = model_from_yaml(f)
    model.load_weights(location + '/weights.h5')
    return model


def add_custom_metric(folder=None, old_way=False, verbose=True):
    """
    I am hardcoding all this for now
    This should only work for datasets similar to radiomics1 (where a 3D volume is divided
    in 3 channel 2D slices)
    """

    (x_train, y_train), (x_test, y_test) = load_dataset("radiomics1")
    patients_train, patients_test = load_patients_dataset("radiomics1")

    # Navigate to folder and load result.yaml
    if folder is not None:
        os.chdir(folder)
    with open("results.yaml") as f:
        try:
            result = yaml.load(f)
        except yaml.YAMLError as YamlError:
            print("There was an error parsing 'results.yaml'. Plotting aborted.")
            print(YamlError)
            if folder is not None:
                os.chdir("./..")
            return

    folders = sorted(result.keys())
    for id in folders:
        print("Folder:", id)
        vol_acc_Tr, vol_acc_Te = calc_vol_acc_Tr_Te(x_train, y_train, x_test, y_test,
                                                    patients_train, patients_test, old_way,
                                                    folder=id)
        result[id]["result"]["volAccTr"] = float(vol_acc_Tr)
        result[id]["result"]["volAccTe"] = float(vol_acc_Te)
        if verbose:
            print("accTr:", result[id]["result"]["accTr"])
            print("accTe:", result[id]["result"]["accTe"])
            print("volAccTr:", result[id]["result"]["volAccTr"])
            print("volAccTe:", result[id]["result"]["volAccTe"])
            print(" ")
    with open("new_results.yaml", "a") as f:
        for location in folders:
            f.write(yaml.dump_all([{location: result[location]}],
                                  default_flow_style=False,
                                  explicit_start=False))

    if folder is not None:
        os.chdir("./..")


def calculate_volume_accuracy(pred_labels, true_labels, pred_percents, observe_training,
                              patients_train, patients_test):
    classification_per_patient = {}
    score_per_patient = {}
    ignore_patient = ""
    if observe_training == 1:
        patients = patients_train
        ignore_patient = patients_test[0]
    elif observe_training == 2:
        patients = patients_train + patients_test
    else:
        patients = patients_test
        ignore_patient = patients_train[-1]
    prev_patient = ""
    new_true_labels = []
    unique_patients = []
    for i, patient in enumerate(patients):
        if patient not in classification_per_patient:
            classification_per_patient[patient] = {}
            score_per_patient[patient] = {}
        try:
            classification_per_patient[patient][pred_labels[i]] += 1
            score_per_patient[patient] += pred_percents[i]
        except KeyError:
            classification_per_patient[patient][pred_labels[i]] = 1
            score_per_patient[patient] = pred_percents[i]
        if prev_patient != patient and patient != ignore_patient:
            new_true_labels.append(true_labels[i])
            unique_patients.append(patient)
        prev_patient = patient

    pred_labels = []
    for patient in unique_patients:
        # Ignore patients that have half the 3D image in test and other half in training
        if patient == ignore_patient:
            continue
        # Assume there are only 2 labels: 0 and 1
        keys = list(classification_per_patient[patient].keys())
        if len(keys) == 1:
            pred_labels.append(keys[0])
        elif len(keys) == 2:
            num_k0 = classification_per_patient[patient][keys[0]]
            num_k1 = classification_per_patient[patient][keys[1]]
            if keys[0] != keys[1]:
                pred_labels.append(keys[0] if num_k0 > num_k1 else keys[1])
            else:
                pred_labels.append(np.argmax(score_per_patient[patient]))
        else:
            print(keys)
            input("This should never happen!")
            continue
    true_labels = np.array(new_true_labels)
    pred_labels = np.array(pred_labels)

    errors_vector = (pred_labels != true_labels)
    num_errors = np.sum(errors_vector)
    size_set = pred_labels.size
    return 1 - num_errors / size_set


def calc_vol_acc_Tr_Te(x_train, y_train, x_test, y_test, patients_train, patients_test, old_way,
                       folder=None, to_categorical=True, data_reduction=None):
    if folder is None:
        folder = "."

    train_set, test_set, input_shape, labels = format_dataset(x_train, y_train, x_test, y_test,
                                                              verbose=False, ret_labels=True,
                                                              data_reduction=data_reduction,
                                                              to_categorical=to_categorical,
                                                              old_way=old_way)
    if data_reduction is not None:
        y_test = y_test[:y_test.shape[0] // data_reduction]
        y_train = y_train[:y_train.shape[0] // data_reduction]

    model = load_model(folder)

    # calculate for training set
    pred_percents = model.predict(train_set[0])
    true_labels = y_train
    pred_labels = np.argmax(pred_percents, axis=1)
    vol_accTr = calculate_volume_accuracy(pred_labels, true_labels, pred_percents, 1,
                                          patients_train, patients_test)

    # calculate for test set
    pred_percents = model.predict(test_set[0])
    true_labels = y_test
    pred_labels = np.argmax(pred_percents, axis=1)
    vol_accTe = calculate_volume_accuracy(pred_labels, true_labels, pred_percents, 0,
                                          patients_train, patients_test)
    return vol_accTr, vol_accTe

if __name__ == "__main__":
    folder = None
    old_way = False
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2].lower() != "false":
        old_way = True
    add_custom_metric(folder=folder, old_way=old_way)

    """
    Expects:
        py add_custom_metric.py
        py add_custom_metric.py folder
        py add_custom_metric.py folder old_way
    """
