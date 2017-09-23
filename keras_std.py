import matplotlib as mpl
# This line allows mpl to run with no DISPLAY defined
mpl.use('Agg')
from matplotlib import pyplot as plt
from keras.callbacks import Callback
from time import clock
import math


# Configure how the callback works only modifying these constants
SHOW_PLOTS = False  # True: plots will be shown, False: plots will be saved

BATCH_PLOT_UPDATE_FREQUENCY = 32                    # num batches between every update in the plot
BATCH_FIG_NUM = 1                                   # figure number where we will show figure
BATCH_SUBFIG_METRICS = ["loss", "acc"]              # metrics to be shown (will be read in on_batch_end)
BATCH_SUBFIG_TITLES = ["Tranining Loss", "Tranining Accuracy"]  # Titles for each metric
BATCH_SUBFIG_YLIM = [3, 1]                          # initial Y range for all subfigures
BATCH_NUM_COLS = 2                                  # number of colums in the figure
BATCH_NUM_ROWS = int(math.ceil(len(BATCH_SUBFIG_METRICS) / BATCH_NUM_COLS))     # number of rows in the figure

EPOCH_PLOT_UPDATE_FREQUENCY = 1                     # num epochs between every plot update. If 0, we only consider EPOCH_PLOT_TIME_LAPSE
EPOCH_PLOT_TIME_LAPSE = 30                          # how often do we plot (every EPOCH_PLOT_TIME_LAPSE seconds)
EPOCH_FIG_NUM = 0                                   # figure number where we will show figure
EPOCH_SUBFIG_METRICS = [["loss", "val_loss"], ["acc", "val_acc"]]   # metrics to be shown (will be read in on_batch_end)
EPOCH_SUBFIG_LABELS = [["training", "validation"], ["training", "validation"]]   # labels for the legend in every subfig
EPOCH_SUBFIG_TITLES = ["Loss", "Accuracy"]          # Titles for each metric
EPOCH_SUBFIG_YLIM = [3, 1]                          # initial Y range for all subfigures
EPOCH_NUM_COLS = 2                                  # number of colums in the figure
EPOCH_NUM_ROWS = int(math.ceil(len(EPOCH_SUBFIG_METRICS) / EPOCH_NUM_COLS))     # number of rows in the figure


class cbPlotEpochBatch(Callback):
    def __init__(self, location=None):
        self.location = location
        if self.location is None:
            self.location = ""
        elif self.location[-1] != "/":
            self.location += "/"

    def on_train_begin(self, logs={}):
        if SHOW_PLOTS:
            plt.ion()  # Allows plots to be non-blocking
        self.init_epoch_plots()
        self.time_plot = clock()

    def on_train_end(self, logs={}):
        self.update_epoch_plots(rescale_Y=True)  # Rescales Y in epoch plots
        if SHOW_PLOTS:
            plt.ioff()  # Make plots blocking again
        else:
            self.epoch_fig.clear()
            try:
                self.batch_fig.clear()
            except AttributeError:
                pass

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch = epoch
        self.init_batch_plots()

    def on_epoch_end(self, epoch, logs={}):
        self.update_batch_plots(rescale_Y=True)  # Rescales Y in batch plots
        self.record_new_epoch_data(logs)

    def on_batch_begin(self, batch, logs={}):
        self.batch = batch

    def on_batch_end(self, batch, logs={}):
        self.record_new_batch_data(logs)

    def init_epoch_plots(self):
        # Every training round create new plot of EPOCH_SUBFIG_METRICS as epochs change
        self.epoch_fig = plt.figure(EPOCH_FIG_NUM)
        self.epoch_fig.clear()
        self.epoch_subfig = []
        self.epoch_metrics = [[] for _ in range(len(EPOCH_SUBFIG_METRICS))]
        self.epoch_lines = [[] for _ in range(len(EPOCH_SUBFIG_METRICS))]
        for i in range(len(EPOCH_SUBFIG_METRICS)):
            self.epoch_subfig.append(
                self.epoch_fig.add_subplot(EPOCH_NUM_ROWS * 100 + EPOCH_NUM_COLS * 10 + i + 1))
            self.epoch_subfig[i].set_ylim(0, EPOCH_SUBFIG_YLIM[i])
            self.epoch_subfig[i].set_title(EPOCH_SUBFIG_TITLES[i])
            self.epoch_subfig[i].set_xlabel("Epoch")
            for j in range(len(EPOCH_SUBFIG_METRICS[i])):
                self.epoch_metrics[i].append([])
                line, = self.epoch_subfig[i].plot([], self.epoch_metrics[i][j],
                                                  label=EPOCH_SUBFIG_LABELS[i][j])
                self.epoch_lines[i].append(line)
            self.epoch_subfig[i].legend()
        self.epoch_num = 0
        self.epoch_steps = []

    def record_new_epoch_data(self, logs):
        for i in range(len(self.epoch_metrics)):
            for j in range(len(self.epoch_metrics[i])):
                self.epoch_metrics[i][j].append(logs.get(EPOCH_SUBFIG_METRICS[i][j]))
        self.epoch_steps.append(self.epoch_num)
        self.epoch_num += 1
        curr_time = clock()
        if ((EPOCH_PLOT_UPDATE_FREQUENCY > 0 and self.epoch_num % EPOCH_PLOT_UPDATE_FREQUENCY == 0) or
                (EPOCH_PLOT_UPDATE_FREQUENCY <= 0 and self.time_plot + EPOCH_PLOT_TIME_LAPSE < curr_time)):
            self.time_plot = curr_time
            self.update_epoch_plots()

    def update_epoch_plots(self, rescale_Y=False):
        for i in range(len(self.epoch_lines)):
            self.epoch_subfig[i].set_xlim(0, self.epoch_num)
            for j in range(len(self.epoch_lines[i])):
                self.epoch_lines[i][j].set_data(self.epoch_steps, self.epoch_metrics[i][j])
        if rescale_Y:  # Rescale ylim so plots fit in every subfigure
            for i in range(len(self.epoch_lines)):
                minY = float("inf")
                maxY = float("-inf")
                for j in range(len(self.epoch_lines[i])):
                    maxY = max(maxY, max(self.epoch_metrics[i][j]))
                    minY = min(minY, min(self.epoch_metrics[i][j]))
                maxY = math.ceil(maxY * 10) / 10
                minY = math.floor(minY * 10) / 10
                rangeY = (maxY - minY) * 0.05
                self.epoch_subfig[i].set_ylim(minY - rangeY, maxY + rangeY)
        if SHOW_PLOTS:
            plt.pause(0.001)
        else:
            self.epoch_fig.savefig(self.location + "loss-acc_epoch.png", bbox_inches="tight")

    def init_batch_plots(self):
        # Every epoch create new plot of BATCH_SUBFIG_METRICS as batches change
        self.batch_fig = plt.figure(BATCH_FIG_NUM)
        self.batch_fig.clear()
        self.batch_subfig = []
        self.batch_metrics = [[] for _ in range(len(BATCH_SUBFIG_METRICS))]
        self.batch_lines = []
        for i in range(len(BATCH_SUBFIG_METRICS)):
            self.batch_subfig.append(
                self.batch_fig.add_subplot(BATCH_NUM_ROWS * 100 + BATCH_NUM_COLS * 10 + i + 1))
            self.batch_subfig[i].set_ylim(0, BATCH_SUBFIG_YLIM[i])
            self.batch_subfig[i].set_title(BATCH_SUBFIG_TITLES[i])
            self.batch_subfig[i].set_xlabel("Batch")
            line, = self.batch_subfig[i].plot([], self.batch_metrics[i])
            self.batch_lines.append(line)
            self.batch_subfig[i].legend()
        self.batch_num = 0
        self.batch_steps = []

    def record_new_batch_data(self, logs):
        for i in range(len(self.batch_metrics)):
            self.batch_metrics[i].append(logs.get(BATCH_SUBFIG_METRICS[i]))
        self.batch_steps.append(self.batch_num)
        self.batch_num += 1
        if self.batch_num % BATCH_PLOT_UPDATE_FREQUENCY == 0:
            self.update_batch_plots()

    def update_batch_plots(self, rescale_Y=False):
        for i in range(len(self.batch_lines)):
            self.batch_subfig[i].set_xlim(0, self.batch_num)
            self.batch_lines[i].set_data(self.batch_steps, self.batch_metrics[i])
        if rescale_Y:  # Rescale ylim so plots fit in every subfigure
            for i in range(len(self.batch_lines)):
                maxY = math.ceil(max(self.batch_metrics[i]) * 10) / 10
                minY = math.floor(min(self.batch_metrics[i]) * 10) / 10
                rangeY = (maxY - minY) * 0.05
                self.batch_subfig[i].set_ylim(minY - rangeY, maxY + rangeY)
        if SHOW_PLOTS:
            plt.pause(0.001)
        else:
            self.batch_fig.savefig(self.location + "loss-acc_batch_epoch{}.png".format(self.epoch), bbox_inches="tight")


class cbPlotEpoch(cbPlotEpochBatch):
    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        self.record_new_epoch_data(logs)

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass