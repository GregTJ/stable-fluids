import numpy as np
from PIL import Image

from fluid import Fluid

FRAME_PATH = 'C:/users/liz/desktop/fluid/'

RESOLUTION = (700, 700)
VISCOSITY = 10 ** -3
DURATION = 200

INFLOW_PADDING = 50
INFLOW_DURATION = 60
INFLOW_RADIUS = 8
INFLOW_VELOCITY = 12


def circle(theta):
    return np.asarray((np.cos(theta), np.sin(theta)))


center = np.asarray(RESOLUTION) // 2
r = np.min(center) - INFLOW_PADDING
directions = tuple(-circle(p * np.pi * 2 / 3) for p in range(3))
points = tuple(r * circle(p * np.pi * 2 / 3) + center for p in range(3))

fluid = Fluid(RESOLUTION, VISCOSITY, ('r', 'g', 'b'))

inflow_dye_field = np.zeros((np.product(RESOLUTION), 3))
inflow_velocity_field = np.zeros_like(fluid.velocity_field)
for i, p in enumerate(points):
    _ = np.dstack(tuple(fluid.indices[..., d] - p[d] for d in range(2)))
    _ = np.linalg.norm(_.squeeze(), axis=1)

    for d in range(2):
        inflow_velocity_field[..., d][_ <= INFLOW_RADIUS] = directions[i][d] * INFLOW_VELOCITY

    inflow_dye_field[..., i][_ <= INFLOW_RADIUS] = 1

for frame in range(DURATION):
    print(f'Computing frame {frame}.')

    fluid.advect_diffuse()

    if frame <= INFLOW_DURATION:
        fluid.velocity_field += inflow_velocity_field

        for i, k in enumerate(('r', 'g', 'b')):
            fluid.quantities[k] += inflow_dye_field[..., i]

    fluid.project()

    rgb = np.dstack((fluid.quantities['r'], fluid.quantities['g'], fluid.quantities['b']))

    rgb = rgb.reshape((*RESOLUTION, 3))
    rgb = (np.clip(rgb, 0, 1) * 255).astype('uint8')
    Image.fromarray(rgb).save(f'{FRAME_PATH}Frame {frame}.png')
