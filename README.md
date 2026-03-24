# JaxAlign

**Fast differentiable image registration library for 2D and 3D images, powered by JAX.**

Built for speed and flexibility using automatic differentiation and native GPU/TPU acceleration.

## Features
- Differentiable geometric transforms (currently Affine)
- Built-in losses (L2 and more)
- JAX-native: JIT, vmap, grad support
- Multi-resolution alignment with sharded batching
- Automatic GPU acceleration — significantly faster than CPU-based tools like DIPY or ANTs
- Clean, minimal API

## Future plans
Non-rigid (deformable) transforms coming soon.

## Installation
```bash
git clone https://github.com/mark-dawn/JaxAlign.git
cd JaxAlign
pip install -e .
```
Requires JAX with GPU/TPU support for best performance.

## Quick Start
```Python
import JaxAlign as jlg
import jax.numpy as jnp
import jax

fixed = jlg.Image(...)   # fixed image + spacing
moving = jlg.Image(...)  # moving image + spacing

layer = jlg.transform.Affine(3)
layer = jlg.align( # TODO: please reference test_align.py
    fixed.mip([2,4,4]),
    moving.mip([1,2,2]),
    layer,
    jlg.loss.L2_loss,
    lr=5e-5,
    steps=45,
    device=jax.devices("gpu")[0]
)

print(layer)
```
Full working example with multi-resolution, timing, loss curve and napari visualization is in test_align.py.

## License
GPL-3.0
