#!/usr/bin/env python3.5

import sys
import os
import yaml
from time import clock
from datetime import datetime, timedelta
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import losses
from keras import optimizers
from keras.datasets import mnist
from keras_utils import format_dataset, flexible_neural_net  # 'Library' by Daniel


def main(data_reduction=None, epochs=100, verbose=False):
    print("Loading training and test sets ...")
    # Load pre-shuffled MNIST data into train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print("Reshaping training and test sets ...")
    train_set, test_set, input_shape, labels = format_dataset(x_train, y_train, x_test, y_test,
                                                              data_reduction=data_reduction, verbose=True,
                                                              to_categorical=True, ret_labels=True)

    now = datetime.now()
    folder = "{}_{:02d}.{:02d}.{:02d}".format(now.date(), now.hour, now.minute, now.second)
    try:
        os.makedirs(folder)
    except OSError:
        pass    # In case the dir already exists
    os.chdir(folder)

    print("Generating and training models...")
    filters1 = [4, 8, 16, 32, 64, 128, 256]
    filters2 = [4, 8, 16, 32, 64, 128, 256]
    optimizers1 = [optimizers.Adadelta(), optimizers.Adagrad(), optimizers.Adam(), optimizers.Adamax(), optimizers.SGD(), optimizers.RMSprop()]
    losses1 = [losses.MSE, losses.MAE, losses.hinge, losses.categorical_crossentropy]
    units1 = [4, 8, 16, 32, 64, 128, 256]
    kernel_sizes = [(3, 3), (5, 5)]
    dropouts1 = [0.25, 0.5, 0.75]
    dropouts2 = [0.25, 0.5, 0.75]
    pool_sizes1 = [(2, 2)]

    filters1 = [16, 32, 64, 128, 256]
    filters2 = [16, 32, 64, 128, 256]
    losses1 = [losses.MSE, losses.MAE, losses.hinge, losses.categorical_crossentropy]
    units1 = [16, 32, 64, 128, 256]
    optimizers1 = [optimizers.Adam()]
    kernel_sizes1 = [(3, 3)]
    dropouts1 = [0.25]
    dropouts2 = [0.5]
    pool_sizes1 = [(2, 2)]

    filters1 = [16, 32, 64]
    filters2 = [16, 32, 64]
    losses1 = [losses.MSE, losses.MAE, losses.hinge, losses.categorical_crossentropy]
    units1 = [16, 32, 64]
    optimizers1 = [optimizers.Adam()]
    kernel_sizes1 = [(3, 3)]
    dropouts1 = [0.25]
    dropouts2 = [0.5]
    pool_sizes1 = [(2, 2)]

    num_iterations = len(filters1) * len(filters2) * len(units1) * len(losses1) * len(optimizers1) * len(kernel_sizes1) * len(pool_sizes1) * len(dropouts1) * len(dropouts2)
    it = 0
    avg_time = 0

    for opt in optimizers1:
        for loss in losses1:
            for f1 in filters1:
                for f2 in filters2:
                    for u1 in units1:
                        for ks in kernel_sizes1:
                            for ps in pool_sizes1:
                                for d1 in dropouts1:
                                    for d2 in dropouts2:
                                        t = clock()
                                        it += 1
                                        print("\nIteration {}/{}".format(it, num_iterations))
                                        try:
                                            loss_txt = str(loss).split()[1]
                                        except:
                                            loss_txt = str(loss)
                                        try:
                                            opt_txt = str(opt).split(".")[2].split()[0]
                                        except:
                                            opt_txt = str(opt)
                                        print("optimizer:     {}".format(opt_txt) +
                                              "\nloss:          {}".format(loss_txt) +
                                              "\nfilter1:       {}".format(f1) +
                                              "\nfilter2:       {}".format(f2) +
                                              "\nunits1:        {}".format(u1) +
                                              "\nkerner_size:   {}".format(ks) +
                                              "\npool_size:     {}".format(ps) +
                                              "\ndropout_rate1: {}".format(d1) +
                                              "\ndropout_rate2: {}".format(d2))

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
                                        params = {"optimizer": opt_txt, "loss": loss_txt,
                                                  "filter1": f1, "filter2": f2, "unit1": u1,
                                                  "kernel_size": ks, "pool_size": ps,
                                                  "dropout_rate1": d1, "dropout_rate2": d2,
                                                  "location": location, "iteration_number": it}
                                        result = {"lossTr": float(lTr), "accTr": float(aTr),
                                                  "lossTe": float(lTe), "accTe": float(aTe),
                                                  "time": float(time)}
                                        with open("results.yaml", "a") as f:
                                            f.write(yaml.dump_all([[{"params": params},
                                                                    {"result": result}]],
                                                                  default_flow_style=False,
                                                                  explicit_start=True))
                                        print("\nResults:  Training:  Loss: {}  Acc: {}".format(lTr, aTr))
                                        print("          Test:      Loss: {}  Acc: {}".format(lTe, aTe))
                                        taken = clock() - t
                                        avg_time += (taken / it)
                                        taken_date = timedelta(seconds=taken)
                                        time_date = timedelta(seconds=time)
                                        left_date = timedelta(seconds=avg_time * (num_iterations - it))
                                        print("          Time taken: {}  (fit & evaluation time: {})".format(taken_date, time_date))
                                        print("          Expected time left: {}".format(left_date))


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
        main(int(sys.argv[1]))
    elif len(sys.argv) == 2:
        main(int(sys.argv[1]), int(sys.argv[2]))
    else:
        main(int(sys.argv[1]), int(sys.argv[2]), True)

    print("\nTotal Time Taken: {} s".format(timedelta(seconds=clock() - t)))
