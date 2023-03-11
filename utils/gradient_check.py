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
