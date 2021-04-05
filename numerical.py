from functools import reduce, partial
from math import factorial

import numpy as np
import scipy.sparse as sp


def difference(derivative, accuracy=1):
    # Implemented based on the article here:
    # http://web.media.mit.edu/~crtaylor/calculator.html
    radius = accuracy + (derivative + 1) // 2 - 1
    points = range(-radius, radius + 1)
    result = np.linalg.inv(np.vander(points))
    return result[-derivative - 1] * factorial(derivative)


def operator(shape, *differences, separate=False):
    def factors():
        for d, coefficients in zip(shape, differences):
            middle = len(coefficients) // 2
            indices = range(-middle, middle + 1)
            diagonals = zip(indices, coefficients)
            diagonals = (c * np.ones(d - abs(i)) for i, c in diagonals)
            yield sp.diags(diagonals, indices)

    if len(differences) == 1:
        differences *= len(shape)

    if separate:
        ids = tuple(sp.identity(d) for d in shape)
        factors = (ids[:i] + (f,) + ids[i + 1:] for i, f in enumerate(factors()))
        _ = partial(sp.kron, format='csc')

        # The sum of these kron product folds is
        # equivalent to the kronsum of the factors.
        # This identity can be derived from the
        # properties of the kronecker product and
        # is analogous to separation of variables.
        # See: https://en.wikipedia.org/wiki/Kronecker_sum_of_discrete_Laplacians
        return tuple(reduce(_, f) for f in factors)

    else:
        # Credit to Philip Zucker for figuring out
        # that kronsum's argument order is reversed.
        # Without that bit of wisdom I'd have lost it.
        _ = partial(sp.kronsum, format='csc')
        return reduce(lambda a, f: _(f, a), factors())
