from functools import reduce
from math import factorial

import numpy as np
import scipy.sparse as sp


def differences(accuracy, order):
    # Implemented based on the article here: http://web.media.mit.edu/~crtaylor/calculator.html
    # By the properties of square Vandermonde matrices this matrix is invertible (non-singular)
    # iff all stencil points are unique. This is always the case for this application.
    def parts(points, o):
        coefficients = np.vander(points)
        coefficients = np.linalg.inv(coefficients)
        return coefficients[-o - 1] * factorial(o)

    return tuple(parts(range(-accuracy, accuracy + 1), o) for o in order)


def matrices(shape, operators, combine):
    def parts():
        for i, o in enumerate(operators):
            diagonals = []
            for j, p in enumerate(o):
                index = j - len(o) // 2
                diagonals.append((p * np.ones(shape[i] - abs(index)), index))

            matrix = sp.diags(*zip(*diagonals))
            if combine:
                yield matrix
            else:
                # The sum of these kronecker product folds is equivalent to the kronecker sum of all the matrices.
                # This identity can be derived from the properties of the kronecker product.
                # This is useful when you need to apply each operator on a different axis,
                # like in the case of finding the divergence of a velocity field using the gradient.
                yield reduce(sp.kron, (matrix if k == i else sp.identity(d) for k, d in enumerate(shape)))

    # Credit to Philip Zucker for figuring out that kronsum's argument order is reversed.
    # Without that bit of wisdom I'd have lost it.
    return reduce(lambda a, b: sp.kronsum(b, a), parts()) if combine else tuple(parts())
