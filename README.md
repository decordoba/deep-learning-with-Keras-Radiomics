# Deep Learning with Keras
Perform a set of experiments with different Neural Network configurations and architectures. The system automatically saves all
the information necessary so that the performance of every experiment can easily be compared to the others.

Several libraries implement plotting methods that allow easy understanding of the experiments performed. Here we can see a sample of the
figures obtained:

<a href="README_images/confusion_matrices.png" title="Confusion matrices"><img src="/README_images/confusion_matrices.png"></a>

<a href="README_images/fig1.png" title="Example of figure comparing the results of different models"><img src="/README_images/fig1.png" width="70%"></a>

<a href="README_images/sample_9s.png" title="Misclassified images for a label (example: MNIST dataset)"><img src="/README_images/sample_9s.png" width="70%" ></a>

<a href="README_images/model.png" title="Detailed schematic of the Keras model used"><img src="/README_images/model.png" width="25%" ></a> <a href="README_images/loss_acc.png" title="Loss and accuracy evolution as epochs go by"><img src="/README_images/loss_acc.png" width="70%" ></a>


## How to use every file

### [easy_experiments_runner.py](easy_experiments_runner.py)
**Start here to see some results without changing any code!**

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

See the following sections to learn how to use every module separately, and to learn to change the architecture used in `easy_experiments_runner.py`.

### [modular_neural_network.py](modular_neural_network.py)

In here, the different architectures that will be tested can be configured. To do so, we need to create a subclass of the `Experiment` class. As our first experiment, we will create an experiment set that tries different combinations of :

```python
class MyFirstExperiment(Experiment):
    def __init__(self):
        # create standard experiments structure
        self.experiments = {"filters1": [16, 32, 64],
                            "filters2": [16, 32, 64],
                            "units1": [16, 32]}

    def run_experiment(self, train_set, test_set, input_shape, labels, comb, epochs):
        # Yes, self.keys_mapper magically works if you defined self.experiments correctly in the __init__ method
        f1 = comb[self.keys_mapper["filters1"]]
        f2 = comb[self.keys_mapper["filters2"]]
        u1 = comb[self.keys_mapper["units1"]]
        # flexible_neural_net will do all the work, it just needs to get the training and test set, the optimizer
        # and loss functions used, and a list of all the keras layers that will be tested.
        return flexible_neural_net(train_set, test_set, optimizers.Adam(), losses.categorical_crossentropy,
                                   # layers in our Sequential model
                                   Conv2D(f1, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
                                   Conv2D(f2, kernel_size=(3, 3), activation='relu'),
                                   MaxPooling2D(pool_size=(2, 2)),
                                   Dropout(0.25),
                                   Flatten(),
                                   Dense(u1, activation='relu'),
                                   Dropout(0.5),
                                   Dense(len(labels), activation='softmax'),
                                   # optional parameters, don't touch them unless you know what you are doing
                                   batch_size=32, epochs=epochs, verbose=False)
```

What happened in the previous class is: the variables that we want to test are saved in `self.experiments` (in this case we will run all the combinations of filters1, filters2 and units1 possible, and each will be an experiment. In this case, we will perform 3x3x2 = 18 experiments). Then, we need to tell in our `run_experiment` function how to apply those parameters, by choosing how a configuration defines the neural network architecture. In the above case, the current combination of parameters is extracted to `f1`, `f2` and `u1` and used to determine the number of units and filters in different layers of our network.

[modular_neural_network.py](modular_neural_network.py) contains several experiments already configured to start playing around, but the configurations and possibilities are endless!

[modular_neural_network.py](modular_neural_network.py) is also an executable file that will run all your experiments and save them in a tree of folders (see the easy_experiments_runner section for more information). Change the experiments and dataset used in the main function to choose the experiments you want to run. Then run the code (it can take many hours to execute) with:

```
$ python3 modular_neural_network.py my_folder_name
```

### [results_plotter.py](results_plotter.py)

This module allows us to visualize the results saved using `modular_neural_network.py` or `easy_experiments_runner.py`, choose our preferred mode of visualization, and save them. To do so, run the following command and the wizard to see and save the figures will be started:

```
$ python3 results_plotter.py my_folder_name
Number of samples: 18

1. filters1
2. filters2
3. units1
Choose the first parameter (X) to plot (type number + ENTER):
>> 1
Parameter X selected: filters1

2. filters2
3. units1
Choose the second parameter (Y) to plot (type number + ENTER) or enter 0 to start over:
>> 3
Parameter Y selected: units1

Position the figure in the preferred perspective, and press ENTER to save it.
Press the Q key + ENTER to skip saving the figure.
```

After selecting the variables that we want to see in the X and Y axis, we will see a figure in a window, and we will be able to rotate it until we get the desired perspective, and decide to save it or skip it, and jump to the next one. With this we will iterate through all the possible combinations of figures with the chosen X and Y axis. The results obtained with the above configuration are as follows:

<a href="README_images/fig2.png" title="filters2=16"><img src="/README_images/fig2.png" width="32%"></a>
<a href="README_images/fig3.png" title="filters2=32"><img src="/README_images/fig3.png" width="32%"></a>
<a href="README_images/fig4.png" title="filters2=64"><img src="/README_images/fig4.png" width="32%"></a>

### [results_observer.py](results_observer.py)

This module allows the user to see the confusion matrix from any model generated using `modular_neural_network.py` or `easy_experiments_runner.py`. Generate it running:

```
$ python3 results_observer.py my_folder_name/nn0000  # Shows confusion matrix
$ python3 results_observer.py my_folder_name/nn0000 confusion_matrix.png  # Shows and saves confusion matrix
```

With this we will get a result like the following:

<a href="README_images/mat1.png" title="Confusion Matrix for test set"><img src="/README_images/mat1.png" width="50%"></a>

It will also load a dialog so see the misclassified examples, in the following way:

```
Welcome to the misclassified images viewer!
Use the number keys + ENTER to select the best option.
Do you want to filter by predicted value or true value?
0. Exit
1. Filter by predicted values
2. Filter by true values
>> 2
Filtering by: True Values

Select the label you want to filter.
 0. Back
 1. Label 0  (80 mistakes)
 2. Label 1  (24 mistakes)
 3. Label 2  (100 mistakes)
 4. Label 3  (112 mistakes)
 5. Label 4  (106 mistakes)
 6. Label 5  (88 mistakes)
 7. Label 6  (64 mistakes)
 8. Label 7  (110 mistakes)
 9. Label 8  (169 mistakes)
10. Label 9  (155 mistakes)
>> ...
```

With this we can obtain results like:
<a href="README_images/sample_9s.png" title="Example of misclassified 9s"><img src="/README_images/sample_9s.png" width="32%"></a>

### [keras_utils.py](keras_utils.py)

### [keras_plot.py](keras_plot.py)

### [keras_std.py](keras_std.py)

### [keras_experiments.py](keras_experiments.py)
