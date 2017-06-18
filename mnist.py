#!/usr/bin/env python3.5

import math
import numpy as np
# plotting
from matplotlib import pyplot as plt
# linear stack of NN layers, perfect for feed-forward CNN
from keras.models import Sequential
# core layers, layers used in almost any neural network
from keras.layers import Dense, Dropout, Activation, Flatten
# CNN layers, convolutional layers to train efficiently on image data
from keras.layers import Conv2D, MaxPooling2D
# Loss functions and optimimizers available
from keras import losses
from keras import optimizers
# utils, used to transoforming the data and plotting
from  keras.utils import np_utils, plot_model
# backend, used to know required data format
from keras import backend
# callbacks, we will use them to modify keras default behavior
from keras.callbacks import Callback
# dataset used: MNIST
from keras.datasets import mnist

# Set constants
PLOT_UPDATE_FREQUENCY = 25  # Every 25 batches, we update the plot

# Set hyperparameters
BATCH_SIZE = 512 #128
EPOCHS = 4  #12


def plot_images(images, labels=None, label_description="Label", labels2=None,
                label2_description="Label", show_errors_only=False, cmap="Greys"):
    """ Show all images in imgs list, waiting for an ENTER to show the next one """
    plt.ion()  # Allows plots to be non-blocking
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
    plt.ioff()  # Make plots blocking again

def plot_weights(w, fig_num=0, title=None, cmap=None):
    """ Show weights of 3D or 4D kernel. Dim0, Dim1: (x, y), Dim2: depth, Dim3: #examples """
    num_imgs = 1
    if w.ndim == 4:
        num_imgs = w.shape[3]
        num_colors = w.shape[2]
        if num_colors < 3:
            w = w[:, :, 0, :]
        elif num_colors > 4:
            print("Too many dimensions, ignoring all bot the first one")
            w = w[:, :, 0, :]
    elif w.ndim == 3:
        num_imgs = w.shape[2]
    NUM_ROWS = math.floor(num_imgs ** 0.5)
    NUM_COLS = math.ceil(num_imgs ** 0.5)
    if NUM_ROWS * NUM_COLS < num_imgs:
        NUM_ROWS += 1
    plt.ion()
    fig = plt.figure(fig_num)
    if title is not None:
        fig.suptitle(title)
    for i in range(num_imgs):
        subfig = fig.add_subplot(NUM_ROWS, NUM_COLS, + i + 1)
        subfig.imshow(w[:, :, i], cmap=cmap)
        subfig.axis('off')
    plt.ioff()


class TrainingHistory(Callback):
    def on_train_begin(self, logs={}):
        plt.ion()  # Allows plots to be non-blocking
        self.figs = [None]
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        self.plot_batch = True

    def on_train_end(self, logs={}):
        # for fig in self.figs:
        #     if fig is not None and fig.canvas.manager.window is not None:
        #         fig.show()
        plt.ioff()  # Make plots blocking again

    def on_epoch_begin(self, epoch, logs={}):
        if self.plot_batch:
            NUM_ROWS = 1
            NUM_COLS = 2
            self.figs.append(plt.figure(len(self.figs)))
            self.subfig = []
            self.metrics = [[] for _ in range(NUM_ROWS * NUM_COLS)]
            self.lines = []
            for i in range(NUM_ROWS * NUM_COLS):
                self.subfig.append(self.figs[-1].add_subplot(NUM_ROWS * 100 + NUM_COLS * 10 + i + 1))
                line, = self.subfig[i].plot([], self.metrics[i])
                self.lines.append(line)
            TITLES = ["Tranining Loss", "Tranining Accuracy"]
            X_LABEL = ["Batch"] * NUM_ROWS * NUM_COLS
            Y_LIM = [3, 1]
            self.losses = self.metrics[0]
            self.accuracies = self.metrics[1]
            self.val_losses = []  # Not plotted
            self.val_accuracies = []  # Not plotted
            for i in range(NUM_ROWS * NUM_COLS):
                self.subfig[i].set_ylim(0, Y_LIM[i])
                self.subfig[i].set_title(TITLES[i])
                self.subfig[i].set_xlabel(X_LABEL[i])
            self.i = 0
            self.steps = []
            self.save_every = PLOT_UPDATE_FREQUENCY
        else:
            self.losses = []
            self.accuracies = []
            self.val_losses = []
            self.val_accuracies = []

    def on_epoch_end(self, epoch, logs={}):
        if self.plot_batch:
            self.update_batch_plots()
        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        self.acc.append(logs.get("acc"))
        self.val_acc.append(logs.get("val_acc"))
        self.update_epoch_plots()
        print("\nLoss    ", self.loss)
        print("Val Loss", self.val_loss)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_acc'))
        # print("Keys", logs.keys())
        if self.plot_batch:
            self.steps.append(self.i)
            self.i += 1
            if self.i % self.save_every == 0:
                self.update_batch_plots()

    def update_batch_plots(self):
        for i in range(len(self.lines)):
            self.subfig[i].set_xlim(0, self.i)
            self.lines[i].set_data(self.steps, self.metrics[i])
        plt.pause(0.001)

    def update_epoch_plots(self):
        if len(self.loss) > 1:
            self.figs[0] = plt.figure(0)
            subfig = self.figs[0].add_subplot(121)
            subfig.set_xlim(0, len(self.loss))
            subfig = self.figs[0].add_subplot(122)
            subfig.set_xlim(0, len(self.acc))
            try:
                self.line_acc.set_data(range(len(self.acc)), self.acc)
                self.line_val_acc.set_data(range(len(self.val_acc)), self.val_acc)
                self.line_loss.set_data(range(len(self.loss)), self.loss)
                self.line_val_loss.set_data(range(len(self.val_loss)), self.val_loss)
            except:
                subfig = self.figs[0].add_subplot(121)
                subfig.set_title("Loss")
                subfig.set_xlabel("Epoch")
                self.line_loss, = subfig.plot(range(len(self.loss)), self.loss, label="training")
                self.line_val_loss, = subfig.plot(range(len(self.val_loss)), self.val_loss,
                                                  label="validation")
                subfig.set_ylim(0, 3)
                subfig.legend(loc='upper left')
                subfig = self.figs[0].add_subplot(122)
                subfig.set_title("Accuracy")
                subfig.set_xlabel("Epoch")
                self.line_acc, = subfig.plot(range(len(self.acc)), self.acc, label="training")
                self.line_val_acc, = subfig.plot(range(len(self.val_acc)), self.val_acc,
                                                 label="validation")
                subfig.set_ylim(0, 1)
                subfig.legend(loc='upper left')
            plt.pause(0.001)

# Fix random seed for reproducibility
# np.random.seed(314159)

record_history = TrainingHistory()

print("Loading training and test sets ...")
# Load pre-shuffled MNIST data into train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reduce number of examples (for real time debugging)
x_train = x_train[:x_train.shape[0] // 1]
y_train = y_train[:y_train.shape[0] // 1]
x_test = x_test[:x_test.shape[0] // 1]
y_test = y_test[:y_test.shape[0] // 1]

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

# Plot training examples (sanity check)
print("Plotting training images ...")
# plot_images(x_train, y_train)

print("Reshaping training and test sets ...")
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
Y_train = np_utils.to_categorical(y_train, N_LABELS)
Y_test = np_utils.to_categorical(y_test, N_LABELS)

print("Training set shape:  {}".format(X_train.shape))
print("Test set shape:      {}".format(X_test.shape))
print("{} existing labels:  {}".format(N_LABELS, labels))

print("Generating model ...")
# Create model and add layers
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # Regularizes model to prevent overfitting
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(N_LABELS, activation='softmax'))

# Compile model (declare loss function and optimizer)
model.compile(loss=losses.categorical_crossentropy,  # losses.mean_squared_error
              optimizer=optimizers.Adam(),  # optimizers.Adam(),  # optimizers.Adadelta(), optimizers.SGD()
              metrics=['accuracy'])

# Saves neural network model as an image
# plot_model(model, to_file='model.png')

print("Training model ...")
# Fit the model to train data
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1,
                    callbacks=[record_history],
                    validation_data=(X_test, Y_test))

# print trained weights of first layer
weights0 = model.layers[0].get_weights()[0]
plot_weights(weights0, fig_num=EPOCHS + 1, title="Weights layer 0")

print("H Loss    ", history.history['loss'])
print("Val Loss", history.history['val_loss'])

# Plot epoch history for accuracy and loss
plt.figure(EPOCHS + 2)
plt.plot(history.history['acc'], label="train")
plt.plot(history.history['val_acc'], label="test")
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'], label="train")
plt.plot(history.history['val_loss'], label="test")
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("Evaluating model ...")
# Evaluate the model on training and test data
train_score = model.evaluate(X_train, Y_train, verbose=0)
test_score = model.evaluate(X_test, Y_test, verbose=0)

print("Train loss:      {:.6f}".format(train_score[0]))
print("Train accuracy:  {:.6f}".format(train_score[1]))
print("Test loss:       {:.6f}".format(test_score[0]))
print("Test accuracy:   {:.6f}".format(test_score[1]))

print ("Predicting test labels ...")
# Calculate most likely label for every test example
pred_test = model.predict(X_test)
label_test = np.argmax(pred_test, axis=1)

# Plot test examples, and see label comparison
show_errors_only = True
print("Plotting {}test images ...".format("incorrectly classified " if show_errors_only else ""))
plot_images(x_test, labels=y_test, labels2=label_test, label2_description="Predicted label",
            show_errors_only=True)