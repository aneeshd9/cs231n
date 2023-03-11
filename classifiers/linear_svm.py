from typing import Tuple
import numpy as np


def svm_loss_naive(W: np.ndarray,
                   X: np.ndarray,
                   y: np.ndarray,
                   reg: float) -> Tuple[np.float64, np.ndarray]:
    '''
    D = 32 * 32 * 3 + 1 = 3073 (1 for bias)
    C = 10 (number of classes)
    X -> [N, D]
    y -> [N, 1]
    W -> [D, C]
    '''
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    dW = np.zeros(W.shape)

    for i in range(num_train):
        # X[i] -> [1, D] => scores -> [1, C]
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            margin = scores[j] - correct_class_score + 1
            if j == y[i]:
                continue
            if margin > 0:
                loss += margin
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i]
    
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW

def svm_loss_vectorized(W: np.ndarray,
                        X: np.ndarray,
                        y: np.ndarray,
                        reg: float) -> Tuple[np.float64, np.ndarray]:
    '''
    D = 32 * 32 * 3 + 1 = 3073 (1 for bias)
    C = 10 (number of classes)
    X -> [N, D]
    y -> [N, 1]
    W -> [D, C]
    '''
    
    num_train = X.shape[0]
    loss = np.float64(0.0)
    dW = np.zeros(W.shape)
    
    # scores -> [N, C]
    scores = X.dot(W)
    # correct_class_scores -> [N, 1]
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, 1)
    # margins -> [N, C]
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)
    
    margins[margins > 0] = 1
    # margins_sum -> [N, 1]
    margins_sum = np.sum(margins, axis=1)
    margins[np.arange(num_train), y] -= margins_sum
    dW = X.T.dot(margins)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW
