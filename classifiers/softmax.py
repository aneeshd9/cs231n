from typing import Tuple
import numpy as np


def softmax_loss_naive(W: np.ndarray,
                       X: np.ndarray,
                       y: np.ndarray,
                       reg: float) -> Tuple[np.float64, np.ndarray]:
    '''
    D = 32 * 32 * 3 + 1 = 3073
    C = 10
    X -> [N, D]
    y -> [N, 1]
    W -> [D, C]
    '''
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        scores_exp = np.exp(scores)
        scores_exp_sum = np.sum(scores_exp)
        loss += -np.log(scores_exp[y[i]]/scores_exp_sum)
        for j in range(num_classes):
            if j == y[i]:
                dW[:, y[i]] += (scores_exp[y[i]] / scores_exp_sum - 1) * X[i]
            else:
                dW[:, j] += (scores_exp[j] / scores_exp_sum) * X[i]
    
    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    return loss, dW

def softmax_loss_vectorized(W: np.ndarray,
                            X: np.ndarray,
                            y: np.ndarray,
                            reg: float) -> Tuple[np.float64, np.ndarray]:
    '''
    D = 32 * 32 * 3 + 1 = 3073
    C = 10
    X -> [N, D]
    y -> [N, 1]
    W -> [D, C]
    '''
    num_train = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    scores_exp = np.exp(scores)
    scores_exp_sum = np.sum(scores_exp, axis=1, keepdims=True)
    softmax_matrix = scores_exp / scores_exp_sum
    loss = -np.log(softmax_matrix[np.arange(num_train), y])
    loss = np.sum(loss) / num_train + reg * np.sum(W * W)

    dW = softmax_matrix
    dW[np.arange(num_train), y] -= 1
    dW = X.T.dot(dW)
    dW = dW / num_train + 2 * reg * W

    return loss, dW
