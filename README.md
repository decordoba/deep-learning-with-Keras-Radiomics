# Deep Learning with Keras
Perform a set of experiments with different Neural Network configurations and architectures. The system automatically saves all
the information necessary so that the performance of every experiment can easily be compared to the others.

Several libraries implement plotting methods that allow easy understanding of the experiments performed. Here we can see a sample of the
figures obtained:

<a href="README_images/confusion_matrices.png" title="Confusion matrices"><img src="/README_images/confusion_matrices.png"></a>

<a href="README_images/fig1.png" title="Example of figure comparing the results of different models"><img src="/README_images/fig1.png" width="70%"></a>

<a href="README_images/sample_9s.png" title="Misclassified images for a label (example: MNIST dataset)"><img src="/README_images/sample_9s.png" width="70%" ></a>

<a href="README_images/model.png" title="Detailed schematic of the Keras model used"><img src="/README_images/model.png" width="25%" ></a> <a href="README_images/loss_acc.png" title="Loss and accuracy evolution as epochs go by"><img src="/README_images/loss_acc.png" width="70%" ></a>


## What you will find 

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

This will run the 18 experiments defined in `modular_neural_network.MyFirstExperimentShort` (see the [modular_neural_network.py](modular_neural_network.py) section below). A folder will be created (named after the current date and time) where all the data from the experiment will be saved, and inside a folder for every one of the experiments will be created and filled with several files:

* A `history.png` image, with the training and evaluation loss and accuracy when training.
* A `model.png` image containing a detailed model (with all the parameters used in the architecture). This model is generated using a modification of the `keras.utils.plot_model` function, which will print more information in the model.
* A `model.yaml` and a `weights.h5` file. These files hold the model used and the weights obtained after the training. With them, we can load the models into Keras again, so all the training is not lost (see .
* A custom `result.yaml` file, which will save the data generated while training (loss and accuracy change, time taken etc.). The information saved there will be used for plotting.

After this, some plots are generated and saved to the above folder (the one named after the current date and time). The program is configured to run until the end without human supervision. For custom plotting and more control on the figures saved, which can be tweaked manually, modify the parameters passed to plot_results.

Finally, all the models in the folders created previously are loaded, and several confusion matrices are generated and saved in their respective folders. It is also possible to see the misclassified examples in a figure (using a wizard that allows you to choose what examples to look at), and modify the aspect of the confusion matrices. To modify these and other settings, modify the parameters passed to observe_results.

As a final note, the [easy_experiments_runner.py](easy_experiments_runner.py) module is simply a process that calls three functions that were created to be used separately: `keras_experiments.experiments_runner`, `results_plotter.plot_results`, `results_observer.observe_results`. Therefore, they can be run sequentially like this:

```
$ python3 modular_neural_network.py
$ python3 results_plotter.py
$ python3 results_observer.py
```

See the following sections to learn how to use every module separately, and to learn to change the architeccture used in `easy_experiments_runner.py`.

### [modular_neural_network.py](modular_neural_network.py)

### [results_observer.py](results_observer.py)

### [results_plotter.py](results_plotter.py)

### [keras_utils.py](keras_utils.py)

### [keras_plot.py](keras_plot.py)

### [keras_std.py](keras_std.py)

### [keras_experiments.py](keras_experiments.py)
