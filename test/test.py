import sys, os, gc, time
sys.path.append('.')
import JaxAlign as jlg
import jax.numpy as jnp
import jax
import numpy as np
from matplotlib import pyplot as plt

jax.config.update("jax_platform_name", "cpu")
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
cpus = jax.devices("cpu")
gpus = jax.devices("gpu")

# TODO move this in the main module
def align(ref, mov, model, loss_fn, lr: float|jlg.transform.Transform=1e-4, steps=50, device=None):
    # prepare
    fixed_shape = ref.img.shape
    block_size = fixed_shape // np.gcd(fixed_shape, np.pow(2,np.arange(8))[:, np.newaxis]).max(axis=0)
    print(block_size)
    mov = jax.device_put(mov, device)
    model = jax.device_put(model, device)

    @jax.value_and_grad
    def imgloss(transform: jlg.transform.Transform, truth: jnp.ndarray, grid: jnp.ndarray):
        pred = mov.sample_at(transform(grid))
        return loss_fn(truth, pred)

    def sharded_loss_fn(transform, truth_s, grid_s):
        # TODO: check
        losses_and_grads = jax.vmap(
            lambda r, g: imgloss(transform, jax.device_put(r, device), jax.device_put(g, device)),
            (0,0), 0
        )(truth_s, grid_s)
        return jax.tree.map(jax.tree_util.Partial(jnp.sum, axis=0), losses_and_grads)

    # TODO: implement different and more efficent update methods like ADAM
    @jax.tree_util.Partial(jax.jit, device=device)
    def update(model, r_img, r_grid,):
        err, grad = sharded_loss_fn(
            model,
            r_img.reshape(-1, *block_size),
            r_grid.reshape(-1, *block_size, 3)
        )
        new_params = jax.tree.map(
            lambda param, g, learn_rate: param - g * learn_rate,
            model, grad, lr
        )
        return new_params, err

    for _ in range(steps):
        model, err = update(model, ref.img, ref.grid, )
        losses.append(jax.device_put(err, cpus[0]))

    return jax.device_put(model, ref.img.device)

if __name__ == '__main__':
    layer = jlg.transform.Affine(3)
    LEARNING_RATE = jlg.transform.Affine(5e-5, 2.5)
    STEPS = 45
    losses = []

    def load(path, spacing):
        img = jax.device_put(jnp.load(path).astype(jnp.float32), cpus[0])
        return  jlg.Image((img-img.mean()) / img.std(), spacing )

    moving = load('/home/utente/Projects/TorchReg/test/gaba_img.npy', jnp.array([1.26, 1.39, 1.39]))
    fixed = load('/home/utente/Projects/TorchReg/test/live.npy', jnp.array([1., .83, .83]))

    start = time.time()
    layer = align(
        fixed.mip([2,4,4]), moving.mip([1,2,2]),
        layer, jlg.loss.L2_loss, lr=LEARNING_RATE, steps=STEPS,
        device=gpus[0]
    )
    # layer = align(fixed, moving, layer, L2_loss, lr=LEARNING_RATE, steps=STEPS, device=gpus[0])
    print(f"step time: {(total:=time.time()-start) / STEPS}, total: {total}")

    gc.collect()
    jax.clear_caches()
    plt.plot(losses)
    plt.show(block=True)
    print(layer)

    from napari import Viewer
    vw = Viewer()
    layer = jax.device_put(layer,cpus[0], )
    pred = jlg.Image(moving.sample_at(layer(fixed.grid)), fixed.spacing)
    print(moving, fixed, pred)
    vw.add_image(moving.img, **moving.napari_metadata, colormap='blue', name='moving')
    vw.add_image(fixed.img, **fixed.napari_metadata, colormap='green', opacity=.5, name='fixed')
    vw.add_image(pred.img, **pred.napari_metadata, colormap='red', opacity=.5, name='transformed')
    vw.show(block=True)
