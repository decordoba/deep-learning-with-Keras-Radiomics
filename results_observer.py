#!/usr/bin/env python3.5

import sys
import numpy as np
from keras.datasets import mnist
from keras.models import model_from_yaml
from keras_plot import plot_images, plot_confusion_matrix  # 'Library' by Daniel
from keras_utils import format_dataset  # 'Library' by Daniel


def load_model(location):
    with open(location + "/model.yaml", "r") as f:
        model = model_from_yaml(f)
    model.load_weights(location + '/weights.h5')
    return model


def observe_results(data_generator, folder=None, to_categorical=True, data_reduction=None):
    if folder is None:
        folder = "."

    print("Loading training and test sets ...")
    (x_train, y_train), (x_test, y_test) = data_generator()

    print("Reshaping training and test sets ...")
    train_set, test_set, input_shape, labels = format_dataset(x_train, y_train, x_test, y_test,
                                                              verbose=True, ret_labels=True,
                                                              data_reduction=data_reduction,
                                                              to_categorical=to_categorical)
    if data_reduction is not None:
        x_test = x_test[:x_test.shape[0] // data_reduction]
        y_test = y_test[:y_test.shape[0] // data_reduction]

    print("Loading model from {} ...".format(folder))
    model = load_model(folder)

    print("Calculating predicted labels ...")
    pred_test = model.predict(test_set[0])
    label_test = np.argmax(pred_test, axis=1)
    errors_vector = (y_test != label_test)
    num_errors = np.sum(errors_vector)
    size_set = label_test.size
    print("Results: {} errors from {} test examples (Accuracy: {})".format(num_errors, size_set,
                                                                           num_errors / size_set))

    print("Drawing confusion matrix ...")
    plot_confusion_matrix(y_test, label_test, labels, filename=None, title="Confusion Matrix")

    errors_indices = np.argwhere(errors_vector)
    errors_by_predicted_label = dict([(label, []) for label in labels])
    errors_by_expected_label = dict([(label, []) for label in labels])

    for idx in errors_indices:
        errors_by_expected_label[y_test[idx][0]].append(idx[0])
        errors_by_predicted_label[label_test[idx][0]].append(idx[0])

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
            print("0. Back")
            for i, key in enumerate(labels):
                print("{}. Label {}".format(i + 1, key))
            num = -1
            while num < 0 or num > len(labels):
                try:
                    num = int(input(">> "))
                except ValueError:
                    num = -1
            if num == 0:
                break
            print("Plotting misclassified examples for {} label {}\n".format("predicted" if pred_notrue else "true",
                                                                             labels[num - 1]))

            if pred_notrue:
                indices = errors_by_predicted_label[labels[num - 1]]
            else:
                indices = errors_by_expected_label[labels[num - 1]]
            plot_images(x_test[indices], labels=y_test[indices], labels2=label_test[indices],
                        label2_description="Predicted label", fig_num=1)


    txt = input("Press ENTER to see all the misclassified examples unsorted one by one, or q to exit.")
    if len(txt) <= 0 or txt[0] != "q":
        # Plot test examples, and see label comparison
        show_errors_only = True
        print("Plotting {}test images ...".format("incorrectly classified " if show_errors_only else ""))
        plot_images(x_test, labels=y_test, labels2=label_test, label2_description="Predicted label",
                    show_errors_only=True, fig_num=1)


if __name__ == "__main__":
    data = mnist.load_data
    folder = None
    if len(sys.argv) > 1:
        folder = sys.argv[1]

    observe_results(data, folder=folder)

    """
    Expects:
        py results_observer.py
        py results_observer.py folder
    """