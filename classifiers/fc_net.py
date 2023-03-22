from typing import Dict, List, Tuple, Union
import numpy as np

from classifiers.layer_utils import affine_relu_backward, affine_relu_forward, generic_backward, generic_forward
from classifiers.layers import affine_backward, affine_forward, softmax_loss


class FullyConnectedNet(object):
    def __init__(self,
                 hidden_dims: List[int],
                 input_dim: int = 3 * 32 * 32,
                 num_classes: int = 10,
                 dropout_keep_ratio: float = 1.,
                 normalization: str | None = None,
                 reg: float = 0.0,
                 weight_scale: float = 1e-2,
                 dtype: type = np.float32,
                 seed: int | None =None) -> None:
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1.0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        for l, (i, j) in enumerate(zip([input_dim, *hidden_dims], [*hidden_dims, num_classes])):
            self.params[f'W{l + 1}'] = np.random.rand(i, j) * weight_scale
            self.params[f'b{l + 1}'] = np.zeros(j)

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = { 'mode': 'train', 'p': dropout_keep_ratio }
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.normalization == 'batchnorm':
            self.bn_params = [{'mode': 'train'} for _ in range(self.num_layers - 1)]
        if self.normalization == 'layernorm':
            self.bn_params = [{} for _ in range(self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray | Tuple[float,
                                                                              Dict[str, np.ndarray]]:
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization == 'batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        cache = {}
        for l in range(self.num_layers):
            keys = [f'W{l + 1}', f'b{l + 1}']
            w, b = (self.params[k] for k in keys)
            X, cache[l] = generic_forward(X, w, b, l==self.num_layers - 1)
        scores = X

        if y is None:
            return scores
        
        loss, grads = 0.0, {}

        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum([np.sum(W ** 2) for k, W in self.params.items() if 'W' in k])

        for l in reversed(range(self.num_layers)):
            dout, dw, db = generic_backward(dout, cache[l])
            grads[f'W{l + 1}'] = dw + self.reg * self.params[f'W{l + 1}']
            grads[f'b{l + 1}'] = db

        return loss, grads


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
