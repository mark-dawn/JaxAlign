from jax import numpy as jnp
import jax.scipy.interpolate as jinterp
from collections import namedtuple
from typing import Tuple, Iterable

class Transform(Iterable):
    def __call__(self, _in):
        return self.forward(_in)

    def forward(self, grid):
        raise NotImplementedError()

class Chain(Transform, Tuple):
    #__slots__ = ('_layers', )
    def __new__(cls, *transforms: Transform):
        return super().__new__(cls, (param for transform in transforms for param in transform))

    def __init__(self, *transforms: Transform):
        self._layers = transforms

    def forward(self, grid):
        for layer in self._layers:
            grid = layer.forward(grid)
        return grid

class Affine(Transform, namedtuple("Affine", ['weight', 'bias'])):
    __slots__ = ()
    def __new__(cls, *args):
        if len(args) == 1:
            weight, bias = cls.make_args(args[0])
        else:
            weight, bias = args
        return super().__new__(cls, weight, bias)

    @staticmethod
    def make_args(ndim):
        return jnp.diag(jnp.ones(ndim)), jnp.zeros(ndim)

    def forward(self, grid):
        return grid @ self.weight + self.bias

    def __repr__(self):
        return f"Affine with bias: {self.bias} and weights: \n {self.weight}"

class VectorField(Transform, namedtuple('VectorField', ['bias'])):
    def __new__(cls, bias):
        if isinstance(bias, (tuple, )):
            bias = jnp.zeros((*bias, 3))
        elif isinstance(bias, (jnp.ndarray)):
            pass
        else:
            raise TypeError("either pass a shape or an initialized jax array")
        return super().__new__(cls, bias)

    def __init__(self, bias)
        gridargs = tuple(
            slice(-(half_width:=shape * step / 2), half_width, shape*1j)
            for shape, step in zip(img.shape, spacing)
        )
        self._interpolator = jinterp.RegularGridInterpolator(
            [ax.flatten() for ax in jnp.ogrid[*gridargs]],
            self.bias,
            fill_value=0.0
        )

    def forward(self, grid):
        return self._interpolator(grid)
