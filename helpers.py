import csv
import os

import numpy as np


def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def f1_score(y_pred, y_true):
    ind = y_true == 1
    tp = sum(y_true[ind] == y_pred[ind])
    fn = sum(y_true[ind] != y_pred[ind])

    ind = y_true == -1
    fp = sum(y_true[ind] != y_pred[ind])

    pr = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * pr * rec / (pr + rec) if (pr + rec) > 0 else 0

    return f1


def load_train_data(data_path, sub_sample=False):
    """
    This function loads the training data and returns the respective numpy arrays.

    Args:
        data_path (str): data folder path
        sub_sample (bool, optional): If True, the data will be subsampled. Default is False.

    Returns:
        x_train (np.array): training data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )

    x_train = x_train[:, 1:]

    # Sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]

    return x_train, y_train


def load_test_data(data_path):
    """
    This function loads the test data and returns the respective numpy arrays.

    Args:
        data_path (str): data folder path

    Returns:
        x_test (np.array): test data
        test_ids (np.array): ids of test data
    """
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    test_ids = x_test[:, 0].astype(dtype=int)
    x_test = x_test[:, 1:]

    return x_test, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def train_test_split(X, y, test_size=0.1, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_val = int(n_samples * test_size)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    return X_train, X_val, y_train, y_val
