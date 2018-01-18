import numpy as np
from importlib import import_module
import pickle


PATH = "data/"


def load_patients_dataset(name, test_percentage=10):
    if name  == "radiomics1" or name == "radiomics2" or name == "radiomics3":
        with open(PATH + name + "/" + name + "_patients.pkl", 'rb') as f:
            patients = pickle.load(f)
        n = int(np.round(len(patients) * (1 - test_percentage / 100)))
        return patients[:n], patients[n:]
    return None


def load_custom_dataset(name, test_percentage=10):
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


def load_dataset(name):
    try:
        # Search name in keras.datasets
        data = getattr(import_module("keras.datasets"), name).load_data  # Only import dataset used
    except AttributeError:
        # Return custom dataset
        data = load_custom_dataset(name)
    return data