import numpy as np
import scipy.sparse as sp
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import factorized

import operators as ops


class Fluid:
    def __init__(self, shape, viscosity, quantities):
        self.shape = shape
        # Defining these here keeps the code somewhat more readable vs. computing them every time they're needed.
        self.size = np.product(shape)
        self.dimensions = len(shape)

        # Variable viscosity, both in time and in space, is easy to set up; but it conflicts with the use of
        # SciPy's factorized function because the diffusion matrix must be recalculated every frame.
        # In order to keep the simulation speedy I use fixed viscosity.
        self.viscosity = viscosity

        # By dynamically creating advected-diffused quantities as needed prototyping becomes much easier.
        self.quantities = {}
        for q in quantities:
            self.quantities[q] = np.zeros(self.size)

        self.velocity_field = np.zeros((self.size, self.dimensions))
        # The reshaping here corresponds to a partial flattening so that self.indices
        # has the same shape as self.velocity_field.
        # This makes calculating the advection map as simple as a single vectorized subtraction each frame.
        self.indices = np.dstack(np.indices(self.shape)).reshape(self.size, self.dimensions)

        self.gradient = ops.matrices(shape, ops.differences(1, (1,) * self.dimensions), False)

        # Both viscosity and pressure equations are just Poisson equations similar to the steady state heat equation.
        laplacian = ops.matrices(shape, ops.differences(1, (2,) * self.dimensions), True)
        self.pressure_solver = factorized(laplacian)
        # Making sure I use the sparse version of the identity function here so I don't cast to a dense matrix.
        self.viscosity_solver = factorized(sp.identity(self.size) - laplacian * viscosity)

    def advect_diffuse(self):
        # Advection is computed backwards in time as described in Jos Stam's Stable Fluids whitepaper.
        advection_map = np.moveaxis(self.indices - self.velocity_field, -1, 0)

        def kernel(field):
            # Credit to Philip Zucker for pointing out the aptness of map_coordinates here.
            # Initially I was using SciPy's griddata function.
            # While both of these functions do essentially the same thing, griddata is much slower.
            advected = map_coordinates(field.reshape(self.shape), advection_map, order=2).flatten()
            return self.viscosity_solver(advected) if self.viscosity > 0 else advected

        # Apply viscosity and advection to each axis of the velocity field and each user-defined quantity.
        for d in range(self.dimensions):
            self.velocity_field[..., d] = kernel(self.velocity_field[..., d])

        for k, q in self.quantities.items():
            self.quantities[k] = kernel(q)

    def project(self):
        # Pressure is calculated from divergence which is in turn calculated from the gradient of the velocity field.
        divergence = sum(self.gradient[d].dot(self.velocity_field[..., d]) for d in range(self.dimensions))
        pressure = self.pressure_solver(divergence)

        for d in range(self.dimensions):
            self.velocity_field[..., d] -= self.gradient[d].dot(pressure)
