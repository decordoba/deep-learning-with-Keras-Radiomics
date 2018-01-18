#!/usr/bin/env python3.5

import sys
import numpy as np
from dataset_loader import load_dataset, load_patients_dataset
from keras.models import model_from_yaml
from keras_plot import plot_images, plot_all_images, plot_confusion_matrix  # 'Library' by Daniel
from keras_utils import format_dataset  # 'Library' by Daniel


def load_model(location):
    with open(location + "/model.yaml", "r") as f:
        model = model_from_yaml(f)
    model.load_weights(location + '/weights.h5')
    return model


def apply_custom_observation(custom_observation):
    # To know how to do this we must look at how the radiomics1 dataset is generated
    # Fortunately (and this is not casual) this dataset generates already a patients.pkl
    # file to help identify what patient each datapoint belongs to, assuming the dataset
    # has not been shuffled
    return load_patients_dataset(custom_observation)



def observe_results(data_generator, folder=None, to_categorical=True, data_reduction=None,
                    mode=0, observe_training=0, filename=None, num_columns=None,
                    misclassified_wizard=True, custom_observation=None):
    """
    :param data_generator: where to get the data from (keras.datasets.mnist.load_data, cifar...)
    :param folder: name of folder where results are found
    :param to_categorical: to_categorical flag when formatting the dataset
    :param data_reduction: if set to a number, use only (1/data_reduction) of all data. None uses all the data
    :param mode: plotting mode, 0 shows color in the main diagonal, 1 does not, 2-3 adds the matrix transposed and only shows lower half of result
    :param observe_training: 0, we observe results for training set, 1 for test set, 2 for both
    :param misclassified_wizard: if True, shows the wizard to see mistakes, else skips this
    :return:
    """
    if folder is None:
        folder = "."

    print("Loading training and test sets ...")
    try:
        # In case data_generator has to be called to get the data
        (x_train, y_train), (x_test, y_test) = data_generator()
    except TypeError:
        # In case data_generator already holds the loaded data (not callable)
        (x_train, y_train), (x_test, y_test) = data_generator

    print("Reshaping training and test sets ...")
    train_set, test_set, input_shape, labels = format_dataset(x_train, y_train, x_test, y_test,
                                                              verbose=True, ret_labels=True,
                                                              data_reduction=data_reduction,
                                                              to_categorical=to_categorical)
    if data_reduction is not None:
        x_test = x_test[:x_test.shape[0] // data_reduction]
        y_test = y_test[:y_test.shape[0] // data_reduction]
        x_train = x_train[:x_train.shape[0] // data_reduction]
        y_train = y_train[:y_train.shape[0] // data_reduction]
        train_set[0] = train_set[0][:train_set[0].shape[0] // data_reduction]
        test_set[0] = test_set[0][:test_set[0].shape[0] // data_reduction]

    print("Loading model from {} ...".format(folder))
    model = load_model(folder)

    print("Calculating predicted labels ...")
    if observe_training == 1:
        pred_percents = model.predict(train_set[0])
        true_labels = y_train
        examples_set = x_train
        confusion_title = "Confusion Matrix (Training Set)"
    elif observe_training == 2:
        pred_percents = model.predict(np.concatenate((train_set[0], test_set[0])))
        true_labels = np.concatenate((y_train, y_test))
        examples_set = np.concatenate((x_train, x_test))
        confusion_title = "Confusion Matrix (Training & Test Set)"
    else:
        pred_percents = model.predict(test_set[0])
        true_labels = y_test
        examples_set = x_test
        confusion_title = "Confusion Matrix (Test Set)"
    pred_labels = np.argmax(pred_percents, axis=1)

    if custom_observation is not None:
        num_errors = sum([abs(x - y) for x, y in list(zip(pred_labels, true_labels))])
        print("Slices Results: {} errors from {} slices (Accuracy: {})".format(num_errors,
                                                                               len(pred_labels),
                                                                               1 - num_errors / len(
                                                                                   pred_labels)))
        confusion_title += " - Custom ({})".format(custom_observation)
        classification_per_patient = {}
        score_per_patient = {}
        patients_train, patients_test = apply_custom_observation(custom_observation)
        ignore_patient = ""
        if observe_training == 1:
            patients = patients_train
        elif observe_training == 2:
            patients = patients_train + patients_test
        else:
            patients = patients_test
            ignore_patient = patients_train[-1]
        prev_patient = ""
        new_true_labels = []
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
            prev_patient = patient

        pred_labels = []
        for patient in classification_per_patient:
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
        print(pred_labels)
        print(new_true_labels)
        true_labels = np.array(new_true_labels)
        pred_labels = np.array(pred_labels)

    errors_vector = (pred_labels != true_labels)
    num_errors = np.sum(errors_vector)
    size_set = pred_labels.size
    print("Results: {} errors from {} examples (Accuracy: {})".format(num_errors, size_set,
                                                                      1 - num_errors / size_set))

    print("Drawing confusion matrix ...")
    ignore_diag = True
    max_scale_factor = 1.0
    color_by_row = False
    half_matrix = False
    if mode == 1 or mode == 3:
        ignore_diag = False
        max_scale_factor = 100.0
        color_by_row = True
        if mode == 3:
            half_matrix = True
    elif mode == 2:
        half_matrix = True
    confusion_mat = plot_confusion_matrix(true_labels, pred_labels, labels,
                                          title=confusion_title, plot_half=half_matrix,
                                          filename=filename, max_scale_factor=max_scale_factor,
                                          ignore_diagonal=ignore_diag, color_by_row=color_by_row)

    print("Counting misclassified examples ...")
    errors_indices = np.argwhere(errors_vector)
    errors_by_predicted_label = dict([(label, []) for label in labels])
    errors_by_expected_label = dict([(label, []) for label in labels])

    for idx in errors_indices:
        print(idx, true_labels[idx][0])
        errors_by_expected_label[true_labels[idx][0]].append(idx[0])
        errors_by_predicted_label[pred_labels[idx][0]].append(idx[0])

    print("Labels that were confused by another value:")
    for i, label in enumerate(labels):
        tp = confusion_mat[i][i]
        fp = len(errors_by_expected_label[label])
        print("    Label {}: {:>3} mistakes, {:>5} right answers => Accuracy: {}".format(label, fp, tp,
                                                                                         tp / (tp + fp)))
    print("Labels that were mistakenly chosen:")
    for i, label in enumerate(labels):
        tp = confusion_mat[i][i]
        fp = len(errors_by_predicted_label[label])
        print("    Label {}: {:>3} mistakes, {:>5} right answers => Accuracy: {}".format(label, fp, tp,
                                                                                         tp / (tp + fp)))

    if not misclassified_wizard:
        return

    while True:
        print("Welcome to the misclassified images viewer!")
        print("Use the number keys + ENTER to select the best option.")
        print("Do you want to filter by predicted value or true value?")
        print("0. Exit\n1. Filter by predicted values\n2. Filter by true values")
        num = -1
        while num < 0 or num > 3:
            try:
                num = int(input(">> "))
            except ValueError:
                num = -1
        if num == 0:
            break
        pred_notrue = num == 1
        print("Filtering by: {} Values\n".format("Predicted" if pred_notrue else "True"))
        while True:
            print("Select the label you want to filter.")
            print("{:>2}. Back".format(0))
            for i, key in enumerate(labels):
                if pred_notrue:
                    num_errors = len(errors_by_predicted_label[key])
                else:
                    num_errors = len(errors_by_expected_label[key])
                print("{:>2}. Label {}  ({} mistakes)".format(i + 1, key, num_errors))
            num = -1
            while num < 0 or num > len(labels):
                try:
                    num = int(input(">> "))
                except ValueError:
                    num = -1
            if num == 0:
                break
            print("Plotting misclassified examples for the {} label {}\n".format("predicted" if pred_notrue else "true",
                                                                                 labels[num - 1]))

            if pred_notrue:
                indices = np.array(errors_by_predicted_label[labels[num - 1]], dtype=int)
                other_labels = true_labels[indices]
                indices = indices[other_labels.argsort()]
                title_labels = true_labels[indices]
                title = "Predicted label: {}".format(labels[num - 1])
            else:
                indices = np.array(errors_by_expected_label[labels[num - 1]], dtype=int)
                other_labels = pred_labels[indices]
                indices = indices[other_labels.argsort()]
                title_labels = pred_labels[indices]
                title = "True label: {}".format(labels[num - 1])
            # plot_images(x_test[indices], labels=y_test[indices], labels2=label_test[indices],
            #             label2_description="Predicted label", fig_num=1)
            plot_all_images(examples_set[indices], labels=title_labels, labels2=None, fig_num=1,
                            suptitle=title, max_cols=num_columns)


    txt = input("Press ENTER to see all the misclassified examples unsorted one by one, or q to exit. ")
    if len(txt) <= 0 or txt[0] != "q":
        # Plot test examples, and see label comparison
        show_errors_only = True
        print("Plotting {}test images ...".format("incorrectly classified " if show_errors_only else ""))
        plot_images(examples_set, labels=true_labels, labels2=pred_labels,
                    label2_description="Predicted label", show_errors_only=True, fig_num=1)


if __name__ == "__main__":
    dataset_name = "radiomics1"  # default dataset used
    custom_observation = "radiomics1"  # set to None to see regular accuracy
    folder = None
    filename = None
    mode = 0
    observe_training = 0
    num_columns = 5
    if len(sys.argv) > 1 and sys.argv[1].lower() != "none":
        folder = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2].lower() != "none":
        filename = sys.argv[2]
    if len(sys.argv) > 3:
        dataset_name = sys.argv[3]
    if len(sys.argv) > 4:
        mode = int(sys.argv[4])
    if len(sys.argv) > 5:
        observe_training = int(sys.argv[5])
    if len(sys.argv) > 6:
        num_columns = int(sys.argv[6])

    data = load_dataset(dataset_name)
    observe_results(data, folder=folder, filename=filename, mode=mode, data_reduction=None,
                    observe_training=observe_training, num_columns=num_columns,
                    custom_observation=custom_observation)

    """
    Expects:
        py results_observer.py
        py results_observer.py folder
        py results_observer.py folder filename dataset_name
        py results_observer.py folder filename dataset_name mode(0-2)
        py results_observer.py folder filename dataset_name mode(0-2) test(0)/training(1)/both(2)
        py results_observer.py folder filename dataset_name mode(0-2) test(0)/training(1)/both(2) num_cols
    """