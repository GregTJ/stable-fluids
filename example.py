import numpy as np
from PIL import Image
from scipy.special import erf

from fluid import Fluid

RESOLUTION = 500, 500
DURATION = 200

INFLOW_PADDING = 50
INFLOW_DURATION = 60
INFLOW_RADIUS = 8
INFLOW_VELOCITY = 20
INFLOW_COUNT = 5


fluid = Fluid(RESOLUTION, 'dye')


def circle(theta):
    return np.asarray((np.cos(theta), np.sin(theta)))


center = np.floor_divide(RESOLUTION, 2)
r = np.min(center) - INFLOW_PADDING
points = tuple(circle(p * np.pi * (1 - (1 / INFLOW_COUNT))) for p in range(INFLOW_COUNT))
normals = tuple(-p for p in points)
points = tuple(r * p + center for p in points)

inflow_velocities = np.zeros_like(fluid.velocity)
inflow_dye = np.zeros(fluid.shape)
for i, p, n in zip(*zip(*enumerate(points)), normals):
    mask = np.linalg.norm(fluid.indices - p.reshape(2, 1, 1), axis=0) <= INFLOW_RADIUS
    inflow_velocities[:, mask] += n.reshape(2, 1)
    inflow_dye[mask] = 1


frames = []
for f in range(DURATION):
    print(f'Computing frame {f}.')
    if f <= INFLOW_DURATION:
        fluid.velocity += inflow_velocities
        fluid.dye += inflow_dye

    curl = fluid.step()[1]
    curl = (erf(curl * 2) + 1) / 4

    color = np.dstack((curl, np.ones(fluid.shape), fluid.dye))
    color = (np.clip(color, 0, 1) * 255).astype('uint8')
    frames.append(Image.fromarray(color, mode='HSV').convert('RGB'))

frames[0].save('example.gif', save_all=True, append_images=frames[1:], duration=20, loop=0)
