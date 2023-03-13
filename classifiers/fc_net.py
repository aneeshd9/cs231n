from typing import Dict, Tuple, Union
import numpy as np

from classifiers.layer_utils import affine_relu_backward, affine_relu_forward
from classifiers.layers import affine_backward, affine_forward, softmax_loss


class TwoLayerNet(object):
    def __init__(self,
                 input_dim: int = 3 * 32 * 32,
                 hidden_dim: int = 100,
                 num_classes: int = 10,
                 weight_scale: float = 1e-3,
                 reg: float = 0.0) -> None:
       self.params = {
               'W1': np.random.randn(input_dim, hidden_dim) * weight_scale,
               'b1': np.zeros(hidden_dim),
               'W2': np.random.randn(hidden_dim, num_classes) * weight_scale,
               'b2': np.zeros(num_classes)
        }

       self.reg = reg

    def loss(self, X: np.ndarray, y: np.ndarray | None = None) -> Union[Tuple[np.float64, Dict[str, np.ndarray]],
                                                          np.ndarray]:
        out, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        out, cache2 = affine_forward(out, self.params['W2'], self.params['b2'])

        if y is None:
            return out
        else:
            loss, dout = softmax_loss(out, y)
            loss += 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))

            dout, dW2, db2 = affine_backward(dout, cache2)
            dout, dW1, db1 = affine_relu_backward(dout, cache1)

            dW1 += self.reg * self.params['W1']
            dW2 += self.reg * self.params['W2']

            grads = { 'W1': dW1, 'W2': dW2, 'b1': db1, 'b2': db2 }

            return loss,  grads
