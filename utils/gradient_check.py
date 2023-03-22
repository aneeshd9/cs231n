from collections.abc import Callable
from random import randrange
import numpy as np


def grad_check_sparse(f: Callable,
                      x: np.ndarray,
                      analytic_grad: np.ndarray,
                      num_checks: int=10, h: float=1e-5):
    '''
    sample a few random elements and only return numerical
    in this dimensions.
    '''

    for _ in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            'numerical: %f analytic: %f, relative error: %e'
            % (grad_numerical, grad_analytic, rel_error)
        )


def eval_numerical_gradient(f: Callable,
                            x: np.ndarray,
                            verbose: bool=True,
                            h: float=0.00001) -> np.ndarray:
    '''
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    '''

    _ = f(x)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) # pyright: ignore
    while not it.finished:
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h
        fxph = f(x)
        x[ix] = oldval - h
        fxmh = f(x)
        x[ix] = oldval
        grad[ix] = (fxph - fxmh) / (2 * h)
        if verbose:
            print(ix, grad[ix])
        it.iternext()

    return grad
