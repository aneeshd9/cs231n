from typing import Tuple
from classifiers.knn import KNearestNeighbor
from datasets.data_utils import load_cifar10

import numpy as np
import os
import matplotlib.pyplot as plt

from utils.time import func_time


def data_setup(num_training: int = -1,
               num_test: int = -1,
               flatten: bool = True
               ) -> Tuple[np.ndarray,
                          np.ndarray,
                          np.ndarray,
                          np.ndarray]:
    cifar10_root = os.path.join(os.path.dirname(__file__),
                                'datasets',
                                'cifar-10-batches-py')

    if not os.path.exists(cifar10_root):
        print('Dataset is not downloaded. Attempting download...')
        try:
            script_dir = os.path.join(os.path.dirname(__file__),
                                      'datasets')
            success = os.system(f'cd {script_dir} && ./get_dataset.sh')
            if success == 0:
                print('Download finished!')
            else:
                raise RuntimeError('Failed to run download script. See terminal output')
        except Exception as e:
            print(e)
            exit(1)

    X_train, y_train, X_test, y_test = load_cifar10(cifar10_root)
    if num_training != -1:
        X_train = X_train[range(num_training)]
        y_train = y_train[range(num_training)]
    if num_test != -1:
        X_test = X_test[range(num_test)]
        y_test = y_test[range(num_test)]
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test

def accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    num_correct = np.sum(pred == gt)
    accuracy = (float(num_correct) / gt.shape[0]) * 100
    print(f'Got {num_correct} / {gt.shape[0]} correct => accuracy {accuracy}')
    return accuracy

def dists_compare(dists_naive: np.ndarray, dists: np.ndarray) -> None:
    difference = np.linalg.norm(dists_naive - dists, ord='fro')
    print(f'The difference was: {difference}')
    if difference < 0.001:
        print('Good! The distance matrices are the same.')
    else:
        print('Uh-oh! The distance matrices are different.')

def cross_validation(X_train: np.ndarray, y_train: np.ndarray) -> int:
    num_folds = 5
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
    X_train_folds = np.array_split(X_train, num_folds)
    y_train_folds = np.array_split(y_train, num_folds)
    k_to_accuracies = {}
    for k in k_choices:
        k_to_accuracies[k] = []
        for fold in range(num_folds):
            classifier = KNearestNeighbor()
            X_train_minus_fold = np.concatenate(X_train_folds[:fold] + X_train_folds[fold + 1:])
            y_train_minus_fold = np.concatenate(y_train_folds[:fold] + y_train_folds[fold + 1:])
            classifier.train(X_train_minus_fold, y_train_minus_fold)
            pred = classifier.predict(X_train_folds[fold], k)
            accuracy = np.mean(pred == y_train_folds[fold])
            k_to_accuracies[k].append(accuracy)

    for k in sorted(k_to_accuracies):
        for accuracy in k_to_accuracies[k]:
            print(f'k = {k}, accuracy = {accuracy}')

    for k in k_choices:
        accuracies = k_to_accuracies[k]
        plt.scatter([k] * len(accuracies), accuracies)

    accuracies_mean = np.array([np.mean(v) for _, v in sorted(k_to_accuracies.items())])
    accuracies_std = np.array([np.std(v) for _, v in sorted(k_to_accuracies.items())])
    plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.show()

    best_k = 1
    max_accuracy = 0.
    for k in sorted(k_to_accuracies):
        max_accuracy_k = max(k_to_accuracies[k])
        if max_accuracy_k > max_accuracy:
            max_accuracy = max_accuracy_k
            best_k = k

    return best_k

def main():
    X_train, y_train, X_test, y_test = data_setup(num_training=5000, num_test=500)
    print(f'Training data shape: {X_train.shape}')
    print(f'Training labels shape: {y_train.shape}')
    print(f'Testing data shape: {X_test.shape}')
    print(f'Testing labels shape: {y_test.shape}')

    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    dists = classifier.compute_distance_two_loops(X_test)
    print(dists.shape)

    y_test_pred = classifier.predict_labels(dists, k=1)
    accuracy(y_test_pred, y_test)

    y_test_pred = classifier.predict_labels(dists, k=5)
    accuracy(y_test_pred, y_test)

    dists_one = classifier.compute_distance_one_loop(X_test)
    dists_compare(dists, dists_one)

    dists_no = classifier.compute_distance_no_loops(X_test)
    dists_compare(dists, dists_no)

    two_loop_time, _ = func_time(classifier.compute_distance_two_loops, X_test)
    print(f'Two loop version took {two_loop_time} seconds.')

    one_loop_time, _ = func_time(classifier.compute_distance_one_loop, X_test)
    print(f'One loop version took {one_loop_time} seconds.')

    no_loop_time, _ = func_time(classifier.compute_distance_no_loops, X_test)
    print(f'No loop version took {no_loop_time} seconds.')

    best_k = cross_validation(X_train, y_train)
    print(f'The best k from cross-validation: {best_k}')

    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    y_test_pred = classifier.predict(X_test, k=best_k)
    accuracy(y_test_pred, y_test)

if __name__ == '__main__':
    main()
