import numpy as np


PATH = "data/"


def load_dataset(name, test_percentage=10):
    if name  == "radiomics1" or name == "radiomics2" or name == "radiomics3":
        f = np.load(PATH + name + "/" + name + ".npz")
        x = f["arr_0"]
        y = f["arr_1"]
        f.close()
        n = int(np.round(len(y) * (1 - test_percentage / 100)))
        x_train = x[:n]
        y_train = y[:n]
        x_test = x[n:]
        y_test = y[n:]
        return (x_train, y_train), (x_test, y_test)
    return None