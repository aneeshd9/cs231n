from six.moves import cPickle as pickle #pyright: ignore

import os
import numpy as np
import matplotlib.pyplot as plt

def load_pickle(f):
    return pickle.load(f, encoding='latin1')

def load_cifar10_batch(filname):
    with open(filname, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        Y = np.array(Y)
        return X, Y

def load_cifar10(root):
    xs = []
    ys = []
    for batch in range(1, 6):
        filename = os.path.join(root, f'data_batch_{batch}')
        X, Y = load_cifar10_batch(filename)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    X_test, y_test = load_cifar10_batch(os.path.join(root, 'test_batch'))
    return X_train, y_train, X_test, y_test

def load_cifar10_metadata(root):
    filename = os.path.join(root, 'batches.meta')
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        return datadict['label_names']

def cifar10(num_training=49000, num_validation=1000,
            num_test=1000, subtract_mean=True):
    cifar10_root = os.path.join(os.path.dirname(__file__),
                                'cifar-10-batches-py')
    X_train, y_train, X_test, y_test = load_cifar10(cifar10_root)
    X_val = X_train[range(num_training, num_training + num_validation)]
    y_val = y_train[range(num_training, num_training + num_validation)]
    X_train = X_train[range(num_training)]
    y_train = y_train[range(num_training)]
    X_test = X_test[range(num_test)]
    y_test = y_test[range(num_test)]

    if subtract_mean:
        mean = np.mean(X_train, axis=0)
        X_train -= mean
        X_val -= mean
        X_test -= mean

    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }

if __name__ == '__main__':
    cifar10_root = os.path.join(os.path.dirname(__file__),
                                'cifar-10-batches-py')
    X_train, y_train, X_test, y_test = load_cifar10(cifar10_root)
    print(f'Training data shape: {X_train.shape}')
    print(f'Training labels shape: {y_train.shape}')
    print(f'Testing data shape: {X_test.shape}')
    print(f'Testing labels shape: {y_test.shape}')

    classes = load_cifar10_metadata(cifar10_root)
    num_classes = len(classes)
    sample_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, sample_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(sample_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

