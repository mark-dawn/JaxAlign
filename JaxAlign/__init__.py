import jax.numpy as jnp
import jax.scipy.interpolate as jinterp
from collections import namedtuple
from . import loss, transform

__all__ = ['loss', 'transform', 'Image']

class Image(namedtuple("Image", ['img', 'spacing'])):
    def __init__(self, img, spacing):
        self.gridargs = tuple(
            slice(-(half_width:=shape * step / 2), half_width, shape*1j)
            for shape, step in zip(img.shape, spacing)
        )
        self._interpolator = jinterp.RegularGridInterpolator(
            [ax.flatten() for ax in jnp.ogrid[*self.gridargs]],
            self.img,
            fill_value=img.min()
        )

    @property
    def grid(self):
        return jnp.mgrid[*self.gridargs].transpose(1,2,3,0)

    def sample_at(self, grid):
        return self._interpolator(grid)

    @property
    def napari_metadata(self):
        return dict(
            scale=self.spacing,
            translate=-jnp.array(self.img.shape)*self.spacing/2
        )

    def __repr__(self) -> str:
        return f"Image {'x'.join(map(str, self.img.shape))}px with {self.spacing} um spacing"

    def mip(self, level: int|list[int]):
        # TODO: add blurring
        if isinstance(level, int):
            levels = (level,) * self.img.ndim
        elif isinstance(level, (list, tuple)) and isinstance(level[0], int):
            levels = tuple(level)
        else:
            raise ValueError("level is either an int or a list of int")
        return Image(self.img[*(slice(None, None, lvl) for lvl in levels)], self.spacing * jnp.array(levels))
