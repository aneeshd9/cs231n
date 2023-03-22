from typing import Tuple
import numpy as np
from classifiers.layers import AffineLayerCache, ReluCache, affine_backward, affine_forward, relu_backward, relu_forward


AffineReluLayerCache = Tuple[AffineLayerCache, ReluCache]
GenericLayerCache = Tuple[
        AffineLayerCache,
        ReluCache
]


def affine_relu_forward(x: np.ndarray,
                        w: np.ndarray,
                        b: np.ndarray) -> Tuple[np.ndarray, AffineReluLayerCache]:
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout: np.ndarray, cache: AffineReluLayerCache) -> Tuple[np.ndarray,
                                                                                 np.ndarray,
                                                                                 np.ndarray]:
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def generic_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray,
                    last: bool = False) -> Tuple[np.ndarray, GenericLayerCache]:
    relu_cache = None
    out, fc_cache = affine_forward(x, w, b)
    if not last:
        out, relu_cache = relu_forward(out)
    cache = (fc_cache, relu_cache)
    return out, cache


def generic_backward(dout: np.ndarray, cache: GenericLayerCache) -> Tuple[np.ndarray,
                                                                          np.ndarray,
                                                                          np.ndarray]:
    fc_cache, relu_cache = cache
    if relu_cache is not None:
        dout = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(dout, fc_cache)
    return dx, dw, db
