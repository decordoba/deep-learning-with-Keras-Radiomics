# Deep Learning with Keras
Perform a set of experiments with different Neural Network configurations and architectures. The system automatically saves all
the information necessary so that the performance of every experiment can easily be compared to the others.

This repository contains several libraries that implement monitoring and plotting methods that allow easy understanding of the experiments performed. The following figures are a sample of the data automatically generated during the experiments:

<a href="README_images/confusion_matrices.png" title="Confusion matrices"><img src="/README_images/confusion_matrices.png"></a>

<a href="README_images/fig1.png" title="Example of figure comparing the results of different models"><img src="/README_images/fig1.png" width="70%"></a>

<a href="README_images/sample_9s.png" title="Misclassified images for a label (example: MNIST dataset)"><img src="/README_images/sample_9s.png" width="70%" ></a>

<a href="README_images/model.png" title="Detailed schematic of the Keras model used"><img src="/README_images/model.png" width="25%" ></a> <a href="README_images/loss_acc.png" title="Loss and accuracy evolution as epochs go by"><img src="/README_images/loss_acc.png" width="70%" ></a>

## Where to start

Start reading the `easy_experiments_runner.py` section to try the program without modifying any code.

## Contents and files

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

This will run the 18 experiments defined in `modular_neural_network.MyFirstExperimentShort` (see the [modular_neural_network.py](modular_neural_network.py) section below). A folder will be created (named after the current date and time) where all the data from the experiment will be stored. Inside this folder, a new folder named `nn[experiment_number]` will be created and filled with the following files for every one of the experiments:

* A `history.png` image, with the training and evaluation loss and accuracy when training.
* A `model.png` image containing a detailed model (with all the parameters used in the architecture). This model is generated using a modification of the `keras.utils.plot_model` function, which will print more information in the model.
* A `model.yaml` and a `weights.h5` file. These files hold the model used and the weights obtained after the training. With them, we can load the models into Keras again, so all the training is not lost (see [documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)).
* A custom `result.yaml` file, which will save the data generated while training (loss and accuracy change, time taken etc.). The information saved there will be used for plotting.

<a href="README_images/folder_tree.png" title="Folder structure"><img src="/README_images/folder_tree.png"></a>

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

    def run_experiment(self, input_shape, labels, comb):
        # Yes, self.keys_mapper magically works if you defined self.experiments correctly in __init__
        f1 = comb[self.keys_mapper["filters1"]]
        f2 = comb[self.keys_mapper["filters2"]]
        u1 = comb[self.keys_mapper["units1"]]
        # return an optimizer and a loss, and a list of layers
        return (optimizers.Adam(), losses.categorical_crossentropy,
                # layers in our Sequential model
                Conv2D(f1, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
                Conv2D(f2, kernel_size=(3, 3), activation='relu'),
                MaxPooling2D(pool_size=(2, 2)),
                Dropout(0.25),
                Flatten(),
                Dense(u1, activation='relu'),
                Dropout(0.5),
                Dense(len(labels), activation='softmax'))
```

What happened in the previous class is: the variables that we want to test are saved in `self.experiments` (in this case we will run all the combinations of filters1, filters2 and units1 possible, and each will be an experiment. In this case, we will perform 3x3x2 = 18 experiments). Then, we need to tell in our `run_experiment` function how to apply those parameters, by choosing how a configuration defines the neural network architecture. In the above case, the current combination of parameters is extracted to `f1`, `f2` and `u1` and used to determine the number of units and filters in different layers of our network.

It is necessary that `run_experiments` returns an optimizer, a loss function and all the layers of the net.

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

It will also load a dialog to see the misclassified examples, in the following way:

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

<a href="README_images/sample_9s.png" title="Example of misclassified 9s"><img src="/README_images/sample_9s.png" width="50%"></a>

### [keras_experiments.py](keras_experiments.py)

Dependency of `modular_neural_network.py` and `easy_experiments_runner.py`, it defines the Experiment abstract class, which will be subclassed to create all the custom experiments. It also contains the `experiments_runner` function:

```python
experiments_runner(data_generator, experiment_obj, folder=None, data_reduction=None, epochs=100,
                   batch_size=32, early_stopping=10, to_categorical=True, verbose=False)
    # Loads the data from data_generator, loads the experiment object used (containing all the
    # experiments that have to be run), creates/opens a folder where it will save all the data,
    # and runs all experiments and saves all results in a folder structure. If the folder already
    # exists and contains old results, the experiments already performed will not be run again.
    # This allows us to stop the execution, and start it again where we left off.
```

This is the main function that `modular_neural_network.py` and `easy_experiments_runner.py` call to save the data of all the experiments. It will be the one in charge of saving all the data in our folder structure

### [keras_utils.py](keras_utils.py)

Contains several useful keras related functions, all of which require to import the TensorFlow backend. These are the main ones:

```python
def format_dataset(x_train, y_train, x_test=None, y_test=None, data_reduction=None,
                   to_categorical=False, ret_labels=False, verbose=False)
    # Reformat input: change dimensions, scale and cast so it can be fed into our model

def flexible_neural_net(train_set, test_set, optimizer, loss, *layers, batch_size=32, epochs=10,
                        callback=cbPlotEpoch, early_stopping=10, location=None, verbose=True)
    # Layer to create a nn model, compile it, fit it, calculate train and test error, and
    # save the model to location with only one function

def save_model_data(model, train_score, test_score, time, location, save_yaml=True, save_json=False,
                    save_image=True, save_weights=True, save_full_model=False, history=None):
    # Save in location some information about the model (location needs to exist!)

def plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True,
               show_params=False)
    # Extension of keras.utils.plot_model
    # Plots model to an image, and also params of layers (original does not do it)
```

`format_dataset` receives a training and test set and formats them so they can be used as input in `flexible_neural_net`. It basically reshapes the input matrix if necessary and makes sure that all dimensions needed exist.

`flexible_neural_net` performs only one experiment. It receives the architecture (including optimizer, loss and layers) as an input, and runs one experiment, saving all the data from such experiment in a `nn[experiment_number]` folder.

`save_model_data` saves the training and test scores (accuracy and loss) and the time taken to run in a `result.yaml` file. It also can save information from an already trained model. What will be saved can be selected using the parameters of the function: the model can be saved in `.yaml` or `.json` format, an image of the model can be saved (`plot_model` will be called), the trained weights can be saved as a `weights.h5` file, the full model can be saved (including weights and architecture) too, and the history can also be saved in the `result.yaml` file.

`plot_model` creates an schematic image of a Sequential model in Keras. This schematic shows the different layers, as well as all the relevant variables that conform each layer, which will vary depending on the layer type. An example of a detailed model schematic can be seen in the following figure:

<a href="README_images/model.png" title="Detailed schematic of a Keras model"><img src="/README_images/model.png" width="40%" ></a>

### [keras_plot.py](keras_plot.py)

Library that uses `matplotlib` to implement several plotting functions:

```
def plot_images(images, fig_num=0, labels=None, label_description="Label", labels2=None,
                label2_description="Label", show_errors_only=False, cmap="Greys", no_axis=True)
    # Show all images in images list, one at a time, waiting for an ENTER to show the next one
    # If q + ENTER is pressed, the function is terminated
    
def plot_all_images(images, fig_num=0, filename=None, labels=None, label_description="Label",
                    labels2=None, label2_description="Label", cmap="Greys", no_axis=True,
                    title=None, max_cols=5)
    # Show several images with labels at the same time (in the same figure)

def plot_weights(w, fig_num=0, filename=None, title=None, cmap=None)
    # Show weights of a 4D kernel [x, y, depth, #kernels]. If w has only 3D, it is assumed that
    # Dim2 is the #kernels, and depth is 1 (B/W kernels). If depths is different to 3 or 4, depth
    # is set to 1, and only the 1st component is used. If filename is None, the figure will be shown,
    # otherwise it will be saved with name filename

def plot_history(history, fig_num=0, filename=None)
    # Plots loss and accuracy in history
    # If filename is None, the figure will be shown, otherwise it will be saved with name filename

def plot_confusion_matrix(true_values, predicted_values, labels, fig_num=0, filename=None,
                          title=None, cmap="plasma", max_scale_factor=100.0, ignore_diagonal=False,
                          color_by_row=False, plot_half=False):
    # Plots a confusion matrix from a list of true and predicted labels. labels must contain the
    # labels names. ignore_diagonal is used to leave the diagonal white (no color), and plot half
    only shows the lower half of the matrix (adding the top and bottom part together).

def plot_confusion_matrix(true_values, predicted_values, labels, fig_num=0, filename=None,
                          title=None, cmap="plasma", max_scale_factor=100.0, ignore_diagonal=False,
                          color_by_row=False, plot_half=False):
    # Plots a confusion matrix from a list of true and predicted labels. The variable labels contains
    # the labels names. ignore_diagonal is used to leave the diagonal white (no color), and plot half
    # only shows the lower half of the matrix (adding the top and bottom part together).
    # max_scale_factor is used to show contrast even for small values in comparison to the larger
    # numbers in the diagonal, set it to 1 to get the right color scale

def plot_3D_bar_graph(X, Y, Z, axis_labels=None, title=None, suptitle=None, filename=None,
                      bars_dist=0.1, fig_num=0, cmap="plasma", view_elev=50, view_azim=45,
                      orthogonal_projection=False, subplot_position=111, fig_clear=True,
                      global_colorbar=False, color_scale=None, figsize=(1, 1), invert_xaxis=False,
                      invert_yaxis=False, zlim=None, save_without_prompt=False):
    """
    Receives list of X, Y and Z and plots them. X and Y can be strings or numbers, Z must be numbers.
    For example:
        plot_3D_bar_graph(["0", "0", "1", "1"], [0, 1, 0, 1], [0, 1, 1, 2])
    will plot a 2 by 2 matrix of bars with different heights.
    Many parameters can be configured, like a title, the labels, a filename to save the figure,
    the distance between the bars, the colormap, the initial point of view...
    :param axis_labels: the label in every axis (x, y, z)
    :param title: the title for the subfigure
    :param suptitle: the global title for the figure
    :param filename: if None, the figure will be plotted, else, the figure will be saved (the user will be prompted with a save console interface)
    :param bars_dist: distance between bars (every bar is a square of side 1-bars_dist in a side 1 grid
    :param fig_num: number of the figure used in matplotlib
    :param cmap: name of colormap used
    :param view_elev: param to determine point of view elevation
    :param view_azim: param to determine point of view rotation
    :param orthogonal_projection: if True, an orthogonal projection is used, else the default oblique is used
    :param subplot_position: indicates the size of the whole figure and the position of the current subfigure (i.e. 122 means figure with 2 subfigs, and we draw in the second one)
    :param fig_clear: whether to clear the whole figure before drawing or not
    :param global_colorbar: whether if the colorbar is global (shared by all subfigures) or local (one for every subfigure)
    :param color_scale: tuple that represents the min and max values used in the scale to draw the colormap. If None, the scale will be picked automatically
    :param figsize: initial size of the figure. By default it is a (1, 1) square, but can be set to (1,2), to change the shape.
    :param invert_xaxis: inverts the xaxis
    :param invert_yaxis: inverts the yaxis
    :param zlim: if not None, sets the scale used in the zaxis, else it is set automatically
    :param save_without_prompt: if True, it will save without showing figure (filename must not be None), else, it shows figure and then it saves it once we press ENTER or cancel with Q
    :return: returns a list of all elevs and azims for all subfigures if filename is not None
    """

def plot_colormap(X, Y, Z, axis_labels=None, title=None, suptitle=None, filename=None, fig_num=0,
                  cmap="plasma", subplot_position=111, fig_clear=True, global_colorbar=False,
                  color_scale=None, figsize=(1, 1), invert_xaxis=False, invert_yaxis=False,
                  save_without_prompt=False):
    """
    Receives list of X, Y and Z and plots them. X and Y can be strings or numbers.
    It will plot a matrix of squares, each with a color representing the number of Z.
    Many parameters can be configured, like a title, the labels, a filename to save the figure,
    the distance between the bars, the colormap, the initial view...
    :param axis_labels: the label in every axis (x, y, z)
    :param title: the title for the subfigure
    :param suptitle: the global title for the figure
    :param filename: if None, the figure will be plotted, else, the figure will be saved (the user will be prompted with a save console interface)
    :param fig_num: number of the figure used in matplotlib
    :param cmap: name of colormap used
    :param subplot_position: indicates the size of the whole figure and the position of the current subfigure (i.e. 122 means figure with 2 subfigs, and we draw in the second one)
    :param fig_clear: whether to clear the whole figure before drawing or not
    :param global_colorbar: whether if the colorbar is global (shared by all subfigures) or local (one for every subfigure)
    :param color_scale: tuple that represents the min and max values used in the scale to draw the colormap. If None, the scale will be picked automatically
    :param figsize: initial size of the figure. By default it is a (1, 1) square, but can be set to (1,2), to change the shape.
    :param invert_xaxis: inverts the xaxis
    :param invert_yaxis: inverts the yaxis
    :param save_without_prompt: if True, it will save without showing figure (filename must not be None), else, it shows figure and then it saves it once we press ENTER or cancel with Q
    """

def plot_graph_grid(X, Y, Z, subaxis_labels=None, axis_labels=None, suptitle=None, filename=None,
                    fig_num=0, scaleX=None, scaleY=None, fig_clear=True, simplified_style=True,
                    legend_label=None, invert_xaxis=False, invert_yaxis=False):
    """
    Receives list of X, Y and Z (Z is a list of lists of points, one for each (X, Y)) and plots them.
    X and Y can be strings or numbers. It basically will print a grid of plots.
    Many parameters can be configured, like a title, the labels, a filename to save the figure,
    the distance between the bars, the colormap, the initial view...
    """
```

### [keras_std.py](keras_std.py)

This library holds the callbacks that create a graph of the loss and accuracy as a neural network is trained. If we add the callback `cbPlotEpoch` when doing `model.fit` in a Sequential Keras model, the loss and accuracy images will be updated every epoch, to allow an easy monitoring of the training. If we add `cbPlotEpochBatch` as a callback to `model.fit`, the loss and accuracy figure will also be updated every several batches. The figures configuration, the update frequency, and other settings can be modified through some constants on top of the `keras_std.py` file.
