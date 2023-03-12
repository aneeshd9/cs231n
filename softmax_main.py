from classifiers.linear_classifier import Softmax
from classifiers.softmax import softmax_loss_naive, softmax_loss_vectorized
from datasets.data_utils import load_cifar10
import numpy as np
import os
import matplotlib.pyplot as plt

from utils.gradient_check import grad_check_sparse
from utils.time import func_time

def tune_hyperparams(X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray) -> Softmax | None:
    results = {}
    best_val = -1
    best_softmax = None

    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]
    combinations = [ (lr, rg) for lr in learning_rates for rg in regularization_strengths]

    for lr, rg in combinations:
        softmax_model = Softmax()
        softmax_model.train(X_train, y_train, learning_rate=lr, reg=rg, num_iters=1000)
        y_train_pred = softmax_model.predict(X_train)
        train_accuracy = np.mean(y_train_pred == y_train)
        y_val_pred = softmax_model.predict(X_val)
        val_accuracy = np.mean(y_val_pred == y_val)
        results[(lr,rg)] = (train_accuracy, val_accuracy)
        if best_val < val_accuracy:
            best_val = val_accuracy
            best_softmax = softmax_model
        
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                    lr, reg, train_accuracy, val_accuracy))
        
    print('Best validation accuracy achieved during cross-validation: %f' % best_val)

    return best_softmax


def data_setup(num_training: int=49000,
               num_validation: int=1000,
               num_test: int=1000,
               num_dev: int=500):
    cifar10_root = os.path.join(os.path.dirname(__file__),
                                'datasets',
                                'cifar-10-batches-py')
    
    X_train, y_train, X_test, y_test = load_cifar10(cifar10_root)
    
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]
    
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

def main():
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = data_setup()
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print('dev data shape: ', X_dev.shape)
    print('dev labels shape: ', y_dev.shape)

    W = np.random.randn(3073, 10) * 0.0001
    loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)
    print(f'Loss: {loss}')
    print(f'Expected loss value: {-np.log(0.1)}')

    f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
    grad_check_sparse(f, W, grad, 10)

    loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
    f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
    grad_check_sparse(f, W, grad, 10)

    naive_time, (naive_loss, naive_grad) = func_time(softmax_loss_naive, W, X_dev, y_dev, 0.000005)
    print(f'Naive loss: {naive_loss} computed in {naive_time}')

    vec_time, (vec_loss, vec_grad) = func_time(softmax_loss_vectorized, W, X_dev, y_dev, 0.000005)
    print(f'Vectorized loss: {vec_loss} computed in {vec_time}')

    grad_difference = np.linalg.norm(naive_grad - vec_grad, ord='fro')
    print(f'Loss difference: {np.abs(naive_loss - vec_loss)}')
    print(f'Gradient difference: {grad_difference}')

    best_softmax = tune_hyperparams(X_train, y_train, X_val, y_val)
    
    if not best_softmax is None:
        y_test_pred = best_softmax.predict(X_test)
        test_accuracy = np.mean(y_test == y_test_pred)
        print('Softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))

        if not best_softmax.W is None:
            w = best_softmax.W[:-1,:]
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
