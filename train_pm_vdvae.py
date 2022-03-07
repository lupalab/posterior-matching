import json
import math
import os
import random
from typing import Dict, Any, Callable, Tuple

import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from absl import app, flags
from bax import TrainState
from bax import Trainer
from bax.callbacks import CheckpointCallback, Callback, LearningRateLoggerCallback
from chex import ArrayTree, Array
from ml_collections.config_flags import config_flags

from posterior_matching.models.vdvae import PosteriorMatchingVDVAE
from posterior_matching.utils import (
    configure_environment,
    load_datasets,
    make_run_dir,
    TensorBoardCallback,
)

configure_environment()

config_flags.DEFINE_config_file("config", lock_config=False)


class ReconstructionCallback(Callback):
    def __init__(
        self,
        reconstruction_fn: Callable[[ArrayTree], Tuple[Array, Array, Array]],
        dataset: tf.data.Dataset,
        num_examples=8,
        seed=None,
    ):
        self._reconstruction_fn = hk.transform(reconstruction_fn).apply

        if jax.local_device_count() > 1:
            ds = dataset.unbatch().batch(
                num_examples // jax.local_device_count(), drop_remainder=True
            )
            ds = ds.batch(jax.local_device_count(), drop_remainder=True)
            self._reconstruction_fn = jax.pmap(self._reconstruction_fn)
        else:
            ds = dataset.unbatch().batch(num_examples, drop_remainder=True)
            self._reconstruction_fn = jax.jit(self._reconstruction_fn)

        self._data_iter = ds.repeat().as_numpy_iterator()

        self._prng = hk.PRNGSequence(seed or random.randint(0, int(2e9)))

    def on_validation_end(
        self, train_state: TrainState, step: int, logs: Dict[str, Any]
    ):
        batch = next(self._data_iter)
        key = self._prng.next()
        if jax.local_device_count() > 1:
            key = jnp.asarray(jax.random.split(key, jax.local_device_count()))
        reconstructions, imputations, samples = self._reconstruction_fn(
            train_state.ema_params, key, batch
        )

        x = batch["image"]
        b = batch["mask"]

        if jax.local_device_count() > 1:
            reconstructions, imputations, samples = jax.device_get(
                (reconstructions, imputations, samples)
            )
            reconstructions = einops.rearrange(reconstructions, "d b ... -> (d b) ...")
            imputations = einops.rearrange(imputations, "d b ... -> (d b) ...")
            samples = einops.rearrange(samples, "d b ... -> (d b) ...")
            x = einops.rearrange(x, "d b ... -> (d b) ...")
            b = einops.rearrange(b, "d b ... -> (d b) ...")

        x_o = np.where(b == 1, x, 127.5)
        reconstruction_images = np.concatenate([x, reconstructions], axis=2).astype(
            np.uint8
        )

        imputations = einops.rearrange(imputations, "b s h w c -> b h (s w) c")
        imputation_images = np.concatenate([x, x_o, imputations], axis=2).astype(
            np.uint8
        )

        sample_images = np.asarray(samples).astype(np.uint8)

        logs["reconstructions"] = reconstruction_images
        logs["imputations"] = imputation_images
        logs["samples"] = sample_images


def main(_):
    config = flags.FLAGS.config

    if "seed" not in config:
        config.seed = random.randint(0, int(2e9))

    config.lock()

    train_dataset, val_dataset = load_datasets(config.data, normalize_images=False)

    def loss_fn(step, is_training, batch):
        model = PosteriorMatchingVDVAE(**config.model)
        out = model(batch["image"], batch["mask"])

        elbo = jnp.mean(out["reconstruction_ll"] - out["kl"])
        del out["reconstruction"]

        out["bpd"] = -elbo / (math.prod(config.model.image_shape) * np.log(2))

        loss = -elbo + jnp.mean(out["pm_kl"])

        return loss, jax.tree_map(jnp.mean, out)

    def reconstruction_fn(batch):
        model = PosteriorMatchingVDVAE(**config.model)
        out = model(batch["image"], batch["mask"])
        imputations = model.impute(batch["image"], batch["mask"], num_samples=8)
        joint_samples = model.sample(num_samples=8)
        return out["reconstruction"], imputations, joint_samples

    warm_up_steps = config.get("warm_up", 0)
    if warm_up_steps > 0:
        schedule = optax.linear_schedule(0, config.lr, warm_up_steps)
    else:
        schedule = lambda _: config.lr

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.scale_by_adam(**config.get("adam", {})),
        optax.add_decayed_weights(
            config.get("weight_decay", 0.0),
            mask=lambda p: jax.tree_map(lambda x: x.ndim != 1, p),
        ),
        optax.scale_by_schedule(schedule),
        optax.scale(-1),
    )

    trainer = Trainer(
        loss_fn,
        optimizer,
        seed=config.seed,
        num_devices=jax.local_device_count(),
        skip_nonfinite_updates=True,
        ema_rate=config.get("ema_rate", 0.999),
        use_ema_for_eval=True,
    )

    run_dir = make_run_dir(prefix=f"pm-vdvae-{config.data.dataset}")
    print("Using run directory:", run_dir)

    with open(os.path.join(run_dir, "model_config.json"), "w") as fp:
        json.dump(config.model.to_dict(), fp)

    callbacks = [
        CheckpointCallback(os.path.join(run_dir, "train_state.pkl")),
        ReconstructionCallback(reconstruction_fn, val_dataset),
        LearningRateLoggerCallback(schedule),
        TensorBoardCallback(os.path.join(run_dir, "tb")),
    ]

    trainer.fit(
        train_dataset,
        config.steps,
        val_dataset=val_dataset,
        validation_freq=config.validation_freq,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    app.run(main)
