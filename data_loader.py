import numpy as np
import math
import random
import os


def norm_plus_minus_1(dataset):
    """Normalize the columns of a dataset between -1 and 1,
    where -1 represents the min value from the columns and 1 is the max.
    """
    m = dataset.shape[1]
    min_cols = np.array([np.min(dataset[:,i]) for i in range(m)])
    max_cols = np.array([np.max(dataset[:,i]) for i in range(m)])
    return -1 + 2*(dataset-min_cols)/(max_cols-min_cols)


def load_dataset(dataset_name):
    """Load a dataset in a 2d array. Split the 2d array in X and y,
    where X is a 2d array containing the inputs of the dataset and
    y are the labels.
    """
    path = os.path.join(os.path.dirname(__file__), 'datasets')
    dataset_name = os.path.join(path, dataset_name)
    dataset = np.genfromtxt(dataset_name, delimiter=',')
    X, y = dataset[:,:-1], dataset[:,-1]
    return X, y


if __name__ == '__main__':
    X, y = load_dataset('ionosphereInput.data')
    print(norm_plus_minus_1(X))
