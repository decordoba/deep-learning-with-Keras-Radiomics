import os
from time import clock
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras import backend
import math
from keras.layers.wrappers import Wrapper
from keras.models import Sequential
from keras_std import cbPlotEpoch  # cbPlotEpochBatch

try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # Fall back on pydot if necessary.
    import pydot


def plot_images(images, fig_num=0, labels=None, label_description="Label", labels2=None,
                label2_description="Label", show_errors_only=False, cmap="Greys"):
    """
    Show all images in images list, one at a time, waiting for an ENTER to show the next one
    If q + ENTER is pressed, the function is terminated
    """
    plt.ion()  # Allows plots to be non-blocking
    fig = plt.figure(fig_num)
    for i, img in enumerate(images):
        if cmap is None:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap=cmap)
        if labels is not None:
            if labels2 is None:
                title = "{} = {}".format(label_description, labels[i])
            else:
                if show_errors_only and labels[i] == labels2[i]:
                    continue
                title = "{} = {} , {} = {}".format(label_description, labels[i],
                                                   label2_description, labels2[i])
            plt.title(title, fontsize="xx-large", fontweight="bold")
        plt.pause(0.001)
        s = input("Press ENTER to see the next image, or Q (q) to continue:  ")
        if len(s) > 0 and s[0].lower() == "q":
            break
    plt.close()  # Hide plotting window
    fig.clear()
    plt.ioff()  # Make plots blocking again

def plot_weights(w, fig_num=0, filename=None, title=None, cmap=None):
    """
    Show weights of a 3D or 4D kernel.
    Dim0, Dim1: (x, y),
    Dim2: depth,
    Dim3: #kernels

    If w has only 3D, it is assumed that Dim2 is the #kernels, and depth is 1 (B/W kernels).
    If depths is different to 3 or 4, depth is set to 1, and only the 1st component is used

    If filename is None, the figure will be shown, otherwise it will be saved with name filename
    """
    num_imgs = 1
    if w.ndim == 4:
        num_imgs = w.shape[3]
        num_colors = w.shape[2]
        if num_colors < 3:
            w = w[:, :, 0, :]
        elif num_colors > 4:
            print("Too many dimensions, ignoring all but the first one")
            w = w[:, :, 0, :]
    elif w.ndim == 3:
        num_imgs = w.shape[2]
    NUM_ROWS = math.floor(num_imgs ** 0.5)
    NUM_COLS = math.ceil(num_imgs ** 0.5)
    if NUM_ROWS * NUM_COLS < num_imgs:
        NUM_ROWS += 1
    if filename is None:
        plt.ion()
    fig = plt.figure(fig_num)
    if title is not None:
        fig.suptitle(title)
    for i in range(num_imgs):
        subfig = fig.add_subplot(NUM_ROWS, NUM_COLS, + i + 1)
        subfig.imshow(w[:, :, i], cmap=cmap)
        subfig.axis('off')
    if filename is None:
        plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()

def plot_history(history, fig_num=0, filename=None):
    """
    Plots loss and accuracy in history
    If filename is None, the figure will be shown, otherwise it will be saved with name filename
    """
    # Plot epoch history for accuracy and loss
    if filename is None:
        plt.ion()
    fig = plt.figure(fig_num)
    subfig = fig.add_subplot(122)
    subfig.plot(history.history['acc'], label="training")
    if history.history['val_acc'] is not None:
        subfig.plot(history.history['val_acc'], label="validation")
    subfig.set_title('Model Accuracy')
    subfig.set_xlabel('Epoch')
    subfig.legend(loc='upper left')
    subfig = fig.add_subplot(121)
    subfig.plot(history.history['loss'], label="training")
    if history.history['val_loss'] is not None:
        subfig.plot(history.history['val_loss'], label="validation")
    subfig.set_title('Model Loss')
    subfig.set_xlabel('Epoch')
    subfig.legend(loc='upper left')
    if filename is None:
        plt.ioff()
    else:
        fig.savefig(filename, bbox_inches="tight")
        fig.clear()

def plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True,
               show_params=False):
    """
    Extension of keras.utils.plot_model
    Plots model to an image, and also params of layers (original does not do it)
    """
    dot = model_to_dot(model, show_shapes, show_layer_names, show_params)
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(to_file, format=extension)

def model_to_dot(model, show_shapes=False, show_layer_names=True, show_params=False):
    """Converts a Keras model to dot format.

    # Arguments
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        show_params: whether to display layer parameteres. (for now only works for some layers)

    # Returns
        A `pydot.Dot` instance representing the Keras model.
    """

    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # Create graph nodes.
    for layer in layers:
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            class_name = '{}({})'.format(class_name, child_class_name)

        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        if show_params:
            if "Conv2D" in class_name:
                label += "|filters: {}\nkernel_size: {}".format(layer.filters, layer.kernel_size)
                label += "\nstrides: {}\npadding: {}".format(layer.strides, layer.padding)
                label += "\nuse_bias: {}\nactivation: {}".format(layer.use_bias,
                                                                 str(layer.activation).split()[1])
                try:
                    label += "\nkernel_reg: {}".format(str(layer.kernel_regularizer).split()[1])
                except IndexError:
                    label += "\nkernel_reg: {}".format(str(layer.kernel_regularizer))
                try:
                    label += "\nbias_reg: {}".format(str(layer.bias_regularizer).split()[1])
                except IndexError:
                    label += "\nbias_reg: {}".format(str(layer.bias_regularizer))
            elif "MaxPooling2D" in class_name or "AveragePooling2D" in class_name:
                label += "|pool_size: {}".format(layer.pool_size)
                label += "\nstrides: {}\npadding: {}".format(layer.strides, layer.padding)
            elif "Dropout" in class_name:
                label += "|rate: {}".format(layer.rate)
            elif "Dense" in class_name:
                label += "|units: {}\nactivation: {}".format(layer.units,
                                                              str(layer.activation).split()[1])
            elif "Activation" in class_name:
                label += "|activation: {}".format(layer.units, str(layer.activation).split()[1])
            elif "BatchNormalization" in class_name:
                try:
                    label += "\ngamma_reg: {}".format(str(layer.gamma_regularizer).split()[1])
                except IndexError:
                    label += "\ngamma_reg: {}".format(str(layer.gamma_regularizer))
                try:
                    label += "\nbeta_reg: {}".format(str(layer.beta_regularizer).split()[1])
                except IndexError:
                    label += "\nbeta_reg: {}".format(str(layer.beta_regularizer))

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
                outputlabels = str(layer.output_shape)
            except AttributeError:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels, outputlabels)

        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot

def format_dataset(x_train, y_train, x_test=None, y_test=None, data_reduction=None,
                   to_categorical=False, ret_labels=False, verbose=False):
    """
    Reformat input: change dimensions, scale and cast so it can be fed into our model
    """
    # Check if x_test is passed
    test_available = True
    if x_test is None:
        x_test = np.array([])
        test_available = False
    if y_test is None:
        y_test = np.array([])
        test_available = False
    # Reduce number of examples (for real time debugging)
    if data_reduction is not None:
        x_train = x_train[:x_train.shape[0] // data_reduction]
        y_train = y_train[:y_train.shape[0] // data_reduction]
        x_test = x_test[:x_test.shape[0] // data_reduction]
        y_test = y_test[:y_test.shape[0] // data_reduction]

    # Get data parameters and save them as 'constants' (they will never change again)
    N_TRAIN = x_train.shape[0]
    N_TEST = x_test.shape[0]
    IMG_ROWS = x_train.shape[1]
    IMG_COLUMNS = x_train.shape[2]
    try:
        IMG_DEPTH = x_train.shape[3]
    except IndexError:
        IMG_DEPTH = 1  # B/W
    labels = np.unique(y_train)
    N_LABELS = len(labels)

    # Reshape input data
    if backend.image_data_format() == 'channels_first':
        X_train = x_train.reshape(N_TRAIN, IMG_DEPTH, IMG_ROWS, IMG_COLUMNS)
        X_test = x_test.reshape(N_TEST, IMG_DEPTH, IMG_ROWS, IMG_COLUMNS)
        input_shape = (IMG_DEPTH, IMG_ROWS, IMG_COLUMNS)
    else:
        X_train = x_train.reshape(N_TRAIN, IMG_ROWS, IMG_COLUMNS, IMG_DEPTH)
        X_test = x_test.reshape(N_TEST, IMG_ROWS, IMG_COLUMNS, IMG_DEPTH)
        input_shape = (IMG_ROWS, IMG_COLUMNS, IMG_DEPTH)

    # Convert data type to float32 and normalize data values to range [0, 1]
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Reshape input labels
    if to_categorical:
        Y_train = np_utils.to_categorical(y_train, N_LABELS)
        Y_test = np_utils.to_categorical(y_test, N_LABELS)

    # Print information about input after reshaping
    if verbose:
        print("Training set shape:  {}".format(X_train.shape))
        print("Test set shape:      {}".format(X_test.shape))
        print("{} existing labels:  {}".format(N_LABELS, labels))

    if ret_labels:
        if not test_available:
            return (X_train, Y_train), input_shape, labels
        return (X_train, Y_train), (X_test, Y_test), input_shape, labels
    if not test_available:
        return (X_train, Y_train), input_shape
    return (X_train, Y_train), (X_test, Y_test), input_shape

def flexible_neural_net(train_set, test_set, optimizer, loss, *layers, batch_size=32, epochs=10,
                        callback=cbPlotEpoch, location=None, verbose=True):
    """
    Layer to create a nn model, compile it, fit it, calculate train and test error, and
    save the model to location with only one function
    """
    # If location is None, create folder with generic name
    if location is None:
        n = 0
        while True:
            location = "nn{:04d}".format(n)
            if not os.path.exists(location):
                break
            n += 1
    try:
        os.makedirs(location)
    except OSError:
        pass  # In case the dir already exists

    # Create model and add layers
    model = Sequential()
    for layer in layers:
        model.add(layer)
    # Compile model (declare loss function and optimizer)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    t = clock()
    # Fit the model to train data
    history = model.fit(train_set[0], train_set[1], batch_size=batch_size, epochs=epochs,
                        verbose=verbose, callbacks=[callback(location)], validation_data=test_set)
    # Evaluate the model on training and test data
    train_score = model.evaluate(train_set[0], train_set[1], verbose=verbose)
    test_score = model.evaluate(test_set[0], test_set[1], verbose=verbose)
    t = clock() - t
    # Save neural network model as an image, and also data
    save_model_data(model, train_score, test_score, t, location, save_weights=True, history=history)
    return train_score, test_score, t

def save_model_data(model, train_score, test_score, time, location, save_yaml=True, save_json=False,
                    save_image=True, save_weights=True, save_full_model=False, history=None):
    """
    Save in location information about the model location needs to exist!)
    """
    if save_image:
        plot_model(model, show_shapes=True, show_layer_names=False, show_params=True,
                   to_file=location + "/model.png")
    if save_yaml:
        with open(location + "/model.yaml", "w") as f:
            f.write(model.to_yaml())  # Load it with model = model_from_json(json_string)
    if save_json:
        with open(location + "/model.json", "w") as f:
            f.write(model.to_json())  # Load it with model = model_from_json(json_string)
    if save_weights:
        model.save_weights(location + '/weights.h5')  # Load it with model.load_weights('w.h5')
    if save_full_model:
        model.save(location + '/model.h5')  # Load it with model = load_model('my_model.h5')
    result = "train_loss: {}\ntrain_accuracy: {}\n".format(train_score[0], train_score[1])
    result += "test_loss: {}\ntest_accuracy: {}\n".format(test_score[0], test_score[1])
    result += "time_taken: {}\n".format(time)
    if history is not None:
        plot_history(history, filename=location + "/history.png")
        result += "train_accuracy: {}\n".format(history.history['acc'])
        result += "train_loss: {}\n".format(history.history['loss'])
        result += "test_accuracy: {}\n".format(history.history['val_acc'])
        result += "test_loss: {}\n".format(history.history['val_loss'])
    with open(location + "/result.yaml", "w") as f:
        f.write(result)

def unpack_layer(object_in_list):
    # Receives a list, where the first param is an object, and the next ones are the
    obj = object_in_list[0]
    general_params = []
    required_params = {}
    for param in object_in_list[1:]:
        if isinstance(param, dict):
            for key in param:
                required_params[key] = param[key]
        else:
            general_params.append(param)
    return obj(*general_params, **required_params)
