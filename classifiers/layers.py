from typing import Tuple
import numpy as np


AffineLayerCache = Tuple[np.ndarray, np.ndarray, np.ndarray]
ReluCache = np.ndarray | None

def affine_forward(x: np.ndarray, w: np.ndarray,
                   b: np.ndarray) -> Tuple[np.ndarray,
                                           AffineLayerCache]:
    '''
    x -> [N, d1, d2, .. ,dk]
    D = d1 * d2 * .. * dk
    w -> [D, M]
    b -> [M, ]

    out -> [N, M]
    '''
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b) return out, cache

def affine_backward(dout: np.ndarray,
                    cache: AffineLayerCache) -> Tuple[np.ndarray,
                                                      np.ndarray,
                                                      np.ndarray]:
    '''
    dout -> [N, M]

    dx -> [N, d1, d2, .. ,dk]
    dw -? [D, M]
    db -> [M, ]
    '''
    x, w, _ = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = dout.T.dot(x.reshape(x.shape[0], -1)).T
    db = dout.sum(axis=0)

    return (dx, dw, db)

def relu_forward(x: np.ndarray) -> Tuple[np.ndarray, ReluCache]:
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout: np.ndarray, cache: ReluCache) -> np.ndarray:
    x = cache
    dx = dout * (x > 0)
    return dx

def svm_loss(x: np.ndarray, y: np.ndarray) -> Tuple[np.float64, np.ndarray]:
    '''
    x -> [N, C] -> scores
    y -> [N, ] -> labels

    loss -> float
    dx -> [N, C]
    '''
    n = x.shape[0]
    
    scores = x
    correct_scores = x[np.arange(n), y].reshape(n, 1)
    margins = np.maximum(0, scores - correct_scores + 1)
    margins[np.arange(n), y] = 0
    loss = np.sum(margins) / n

    margins[margins > 0] = 1
    margins_sum = np.sum(margins, axis=1)
    margins[np.arange(n), y] -= margins_sum
    dx = margins / n

    return loss, dx

def softmax_loss(x: np.ndarray, y: np.ndarray) -> Tuple[np.float64, np.ndarray]:
    '''
    x -> [N, C]
    y -> [N, ]

    loss -> float
    dx -> [N, C]
    '''
    n = x.shape[0]

    scores_exp = np.exp(x)
    scores_exp_sum = np.sum(scores_exp, axis=1, keepdims=True)
    softmax_matrix = scores_exp / scores_exp_sum
    loss = np.sum(-np.log(softmax_matrix[np.arange(n), y])) / n

    softmax_matrix[np.arange(n), y] -= 1
    dx = softmax_matrix / n

    return loss, dx
