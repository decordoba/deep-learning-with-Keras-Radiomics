# Deep Learning with Keras
Perform a set of experiments with different Neural Network configurations and architectures. The system automatically saves all
the information necessary so that the performance of every experiment can easily be compared to the others.

Several libraries implement plotting methods that allow easy understanding of the experiments performed. Here we can see a sample of the
figures obtained:

<a href="README_images/confusion_matrices.png" title="Confusion matrices"><img src="/README_images/confusion_matrices.png"></a>

<a href="README_images/fig1.png" title="Example of figure comparing the results of different models"><img src="/README_images/fig1.png" width="70%"></a>

<a href="README_images/sample_9s.png" title="Misclassified images for a label (example: MNIST dataset)"><img src="/README_images/sample_9s.png" width="70%" ></a>

<a href="README_images/model.png" title="Detailed schematic of the Keras model used"><img src="/README_images/model.png" width="25%" ></a> <a href="README_images/loss_acc.png" title="Loss and accuracy evolution as epochs go by"><img src="/README_images/loss_acc.png" width="70%" ></a>


## Files

### [easy_experiments_runner.py](easy_experiments_runner.py)
Start here to see some results without changing any code!

Open a command terminal, and run the following:

```
$ git clone https://github.com/decordoba/deep-learning-with-Keras.git
$ cd deep-learning-with-Keras/
$ python3 easy_experiments_runner.py
```

This will take several hours to complete (or days). The default behavior tests 108 different neural network architectures sequentially on the mnist dataset. Fortunately, some of the configuration can be modified through command line:

```
$ python3 easy_experiments_runner.py -h
usage: easy_experiments_runner.py [-h] [-d {mnist,cifar10,cifar100}]
                                  [-e {0,1,2,3}] [-f FOLDER]
                                  [-ne NUMBER_EPOCHS] [-dr DATA_REDUCTION]

Create and run experiments with modular_keras, and save results neatly

optional arguments:
  -h, --help            show this help message and exit
  -d {mnist,cifar10,cifar100}, --dataset {mnist,cifar10,cifar100}
                        Name of the dataset to try. Default is 'mnist'.
  -e {0,1,2,3}, --experiment {0,1,2,3}
                        Experiement architecture (0 to 3). Default is 0.
  -f FOLDER, --folder FOLDER
                        Name of the folder where the results are saved. If not
                        set, the folder is named with the current date & time.
  -ne NUMBER_EPOCHS, --number_epochs NUMBER_EPOCHS
                        Maximum number of epochs before termination. Default
                        is 100.
  -dr DATA_REDUCTION, --data_reduction DATA_REDUCTION
                        Number by which to divide the data used. For example,
                        dr=3 means only 1/3 of the data is used. Default is 1.
```

Therefore, the program can be sped-up (by dropping 99% of the data and changing the experiment configuration to a smaller number of architectures) to easily test the program. Just run:

```
$ python3 easy_experiments_runner.py --data_reduction 100 --experiment 3
```

This will run the 18 experiments defined in `modular_neural_network.MyFirstExperimentShort` (see the [modular_neural_network.py](modular_neural_network.py) section below).

### [modular_neural_network.py](modular_neural_network.py)

### [results_observer.py](results_observer.py)

### [results_plotter.py](results_plotter.py)

### [keras_utils.py](keras_utils.py)

### [keras_plot.py](keras_plot.py)

### [keras_std.py](keras_std.py)

### [keras_experiments.py](keras_experiments.py)
