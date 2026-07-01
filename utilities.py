import h5py
import numpy as np

def load_data():
    """
    Loads the cat/dog image dataset from HDF5 files.

    Expects two files in the datasets/ folder:
        trainset.hdf5 — keys: 'X_train' (images), 'Y_train' (labels)
        testset.hdf5  — keys: 'X_test'  (images), 'Y_test'  (labels)

    Returns:
        X_train -- training images  (n_train, 64, 64), pixel values 0-255
        y_train -- training labels  (n_train, 1),      0 = cat, 1 = dog
        X_test  -- test images      (n_test,  64, 64), pixel values 0-255
        y_test  -- test labels      (n_test,  1),      0 = cat, 1 = dog
    """
    train_dataset = h5py.File('datasets/trainset.hdf5', 'r')
    X_train       = np.array(train_dataset['X_train'][:])
    y_train       = np.array(train_dataset['Y_train'][:]).reshape(-1, 1)  # (n,) → (n, 1)

    test_dataset  = h5py.File('datasets/testset.hdf5', 'r')
    X_test        = np.array(test_dataset['X_test'][:])
    y_test        = np.array(test_dataset['Y_test'][:]).reshape(-1, 1)    # (n,) → (n, 1)

    return X_train, y_train, X_test, y_test