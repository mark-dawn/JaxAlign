from jax import numpy as jnp

def L2_loss(truth: jnp.ndarray, pred: jnp.ndarray):
    return jnp.power(pred - truth, 2).mean()

def L1_loss(truth: jnp.ndarray, pred: jnp.ndarray):
    return jnp.abs(pred - truth).mean()
