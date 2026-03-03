from jax import numpy as jnp
from collections import namedtuple
from typing import Tuple, Iterable
from .image import Image

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

class VectorField(Transform, Image, ):
    def __new__(cls, *args):
        if len(args) > 1  and isinstance(args[0], tuple) and isinstance(args[1], tuple):
            bias = jnp.zeros((*args[0], len(args[0])))
            spacing = args[-1]
        elif len(args) == 1 and isinstance(args[0], Image):
            bias = args[0].img
            spacing = args[0].spacing
        else:
            raise TypeError("either pass a shape and spacing or an initialized jalign Image")
        print(bias, spacing)
        return super().__new__(cls, bias, spacing)

    def __init__(self, *args):
        super().__init__(self.img, self.spacing)

    def forward(self, grid):
        return self.sample_at(grid) + grid
