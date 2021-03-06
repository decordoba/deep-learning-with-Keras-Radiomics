import numpy as np
from importlib import import_module
import pickle


PATH = "data/"


def get_end_pos_patient(num=-10, patients=None, name="radiomics2"):
    if patients is None:
        if name.startswith("radiomics_cut"):
            with open(PATH + "radiomics_cut1" + "/" + "radiomics_cut1" + "_patients.pkl", 'rb') as f:
                patients = pickle.load(f)
        elif name.startswith("radiomics_margincut"):
            with open(PATH + "radiomics_marigincut1" + "/" + "radiomics_margincut1" + "_patients.pkl", 'rb') as f:
                patients = pickle.load(f)
        elif name.startswith("radiomics"):
            with open(PATH + "radiomics1" + "/" + "radiomics1" + "_patients.pkl", 'rb') as f:
                patients = pickle.load(f)
    unique_patients = np.unique(patients)
    if num < 0:
        num = len(unique_patients) + num
    prev = ""
    c = 0
    for i, p in enumerate(patients):
        if p != prev:
            c += 1
            if c >= num:
                break
        prev = p
    return i


def load_patients_dataset(name):
    if name == "radiomics1" or name == "radiomics2" or name == "radiomics3":
        with open(PATH + "radiomics1" + "/" + "radiomics1" + "_patients.pkl", 'rb') as f:
            patients = pickle.load(f)
        if name == "radiomics1":
            test_percentage = 10
            n = int(np.round(len(patients) * (1 - test_percentage / 100)))
        elif name == "radiomics2":
            n = get_end_pos_patient(-11, patients)
        else:
            print("Not implemented yet!")
        return patients[:n], patients[n:]
    elif name.startswith("radiomics_cut") or name.startswith("radiomics_margincut"):
        if name.startswith("radiomics_cut"):
            with open(PATH + "radiomics_cut1" + "/" + "radiomics_cut1" + "_patients.pkl", 'rb') as f:
                patients = pickle.load(f)
        else:
            with open(PATH + "radiomics_margincut1" + "/" + "radiomics_margincut1" + "_patients.pkl", 'rb') as f:
                patients = pickle.load(f)
        if name.endswith("1"):
            test_percentage = 10
            n = int(np.round(len(patients) * (1 - test_percentage / 100)))
        elif name.endswith("2"):
            n = get_end_pos_patient(-11, patients)
        else:
            print("Not implemented yet!")
        return patients[:n], patients[n:]
    return None


def load_custom_dataset(name):
    if name == "radiomics1" or name == "radiomics2" or name == "radiomics3":
        f = np.load(PATH + "radiomics1" + "/" + "radiomics1" + ".npz")
        try:
            x = f["x"]
            y = f["y"]
        except KeyError:
            x = f["arr_0"]
            y = f["arr_1"]
        f.close()
        if name == "radiomics1":
            test_percentage = 10
            n = int(np.round(len(y) * (1 - test_percentage / 100)))
        elif name == "radiomics2":
            n = get_end_pos_patient(-11)
        else:
            print("Not implemented yet!")
        x_train = x[:n]
        y_train = y[:n]
        x_test = x[n:]
        y_test = y[n:]
        return (x_train, y_train), (x_test, y_test)
    elif name.startswith("radiomics_cut") or name.startswith("radiomics_margincut"):
        if name.startswith("radiomics_cut"):
            f = np.load(PATH + "radiomics_cut1" + "/" + "radiomics_cut1" + ".npz")
        else:
            f = np.load(PATH + "radiomics_margincut1" + "/" + "radiomics_margincut1" + ".npz")
        try:
            x = f["x"]
            y = f["y"]
        except KeyError:
            x = f["arr_0"]
            y = f["arr_1"]
        f.close()
        if name[-1] == "1":
            test_percentage = 10
            n = int(np.round(len(y) * (1 - test_percentage / 100)))
        elif name[-1] == "2":
            n = get_end_pos_patient(-11)
        else:
            print("Not implemented yet!")
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
