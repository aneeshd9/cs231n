from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np

from classifiers.linear_svm import svm_loss_vectorized


class LinearClassifier(ABC):
    def __init__(self) -> None:
        self.W = None

    def train(self,
              X: np.ndarray,
              y: np.ndarray,
              learning_rate: float=1e-3,
              reg: float=1e-5,
              num_iters: int=100,
              batch_size: int=200,
              verbose: bool=False) -> List[float]:
        num_train, dim = X.shape
        num_classes = np.max(y) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iters):
            idxs = np.random.choice(num_train, batch_size)
            X_batch = X[idxs]
            y_batch = y[idxs]
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W -= grad * learning_rate

            if verbose and it % 100 == 0:
                print(f'Iteration {it}/{num_iters}: loss = {loss}')

        return loss_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.W is None:
            y_pred = np.argmax(X.dot(self.W), axis=1)
            return y_pred
        else:
            raise RuntimeError('Weights of the model have not been initialized.')
    
    @abstractmethod
    def loss(self, X_batch: np.ndarray,
             y_batch: np.ndarray, reg: float) -> Tuple[float, np.ndarray]:
        pass


class LinearSVM(LinearClassifier):
    def loss(self, X_batch: np.ndarray, y_batch: np.ndarray, reg: float) -> Tuple[np.float64, np.ndarray]:
        if not self.W is None:
            return svm_loss_vectorized(self.W, X_batch, y_batch, reg)
        else:
            raise RuntimeError('Weights of the model have not been initialized.')
