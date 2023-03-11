from typing import Tuple
from classifiers.linear_classifier import LinearSVM
from classifiers.linear_svm import svm_loss_naive, svm_loss_vectorized
from datasets.data_utils import load_cifar10
from utils.gradient_check import grad_check_sparse
from utils.time import func_time
import numpy as np
import os
import matplotlib.pyplot as plt


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

def tune_hyperparams(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> LinearSVM | None:
    results = {}
    best_val = -1
    best_svm = None

    learning_rates = [1e-7, 5e-5]
    reg = [2.5e4, 5e4]
    combinations = [(lr, rg) for lr in learning_rates for rg in reg]

    for lr, rg in combinations:
        svm = LinearSVM()
        svm.train(X_train, y_train, lr, rg, 1500, verbose=True)
        y_train_pred = svm.predict(X_train)
        train_accuracy = np.mean(y_train_pred == y_train)
        y_val_pred = svm.predict(X_val)
        val_accuracy = np.mean(y_val_pred == y_val)
        results[(lr, rg)] = (train_accuracy, val_accuracy)

        if best_val < val_accuracy:
            best_val = val_accuracy
            best_svm = svm

    for lr, rg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, rg)]
        print(f'lr {lr} rg {rg} train accuracy: {train_accuracy}, val_accuracy: {val_accuracy}')

    print(f'Best validation accuracy achieved: {best_val}')

    return best_svm

def main():
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

    X_train, y_train, X_test, y_test = data_setup()
    
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Dev data shape: ', X_dev.shape)
    print('Dev labels shape: ', y_dev.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)

    W = np.random.randn(3073, 10) * 0.0001
    
    _, grad = svm_loss_naive(W, X_dev, y_dev, 0.0)
    f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
    grad_check_sparse(f, W, grad)

    naive_loss_time, (loss_naive, grad_naive) = func_time(svm_loss_naive, W, X_dev, y_dev, 0.0005)
    print(f'Naive loss: {loss_naive} and grad: {grad_naive} calculated in {naive_loss_time}')

    vectorized_loss_time, (loss_vectorized, grad_vectorized) = func_time(svm_loss_vectorized, W, X_dev, y_dev, 0.0005)
    print(f'Vectorized loss: {loss_vectorized} and grad: {grad_vectorized} calculated in {vectorized_loss_time}')
    
    norm_ord = 'fro'
    print(f'Loss difference: {loss_naive - loss_vectorized}')
    print(f'Grad difference: {np.linalg.norm(grad_naive - grad_vectorized, ord=norm_ord)}')

    svm = LinearSVM()
    train_time, loss_hist = func_time(svm.train, X_train, y_train, 1e-7,
                                      2.5e4, 1500, 200, True)
    print(f'That took {train_time}s')

    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    y_train_pred = svm.predict(X_train)
    print(f'Training accuracy: {np.mean(y_train == y_train_pred)}')
    y_val_pred = svm.predict(X_val)
    print(f'Validation accuracy: {np.mean(y_val == y_val_pred)}')

    best_svm = tune_hyperparams(X_train, y_train, X_val, y_val)
    if not best_svm is None:
        y_test_pred = best_svm.predict(X_test)
        test_accuracy = np.mean(y_test_pred == y_test)
        print(f'Linear SVM on raw pixels final test set accuracy: {test_accuracy}')
        
        if not best_svm.W is None:
            w = best_svm.W[:-1,:]
            w = w.reshape(32, 32, 3, 10)
            w_min, w_max = np.min(w), np.max(w)
            classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            for i in range(10):
                plt.subplot(2, 5, i + 1)
                  
                wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
                plt.imshow(wimg.astype('uint8'))
                plt.axis('off')
                plt.title(classes[i])  
            plt.show()

if __name__ == '__main__':
    main() 
