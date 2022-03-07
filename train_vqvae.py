import json
import os
import random
from typing import Dict, Any, Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from absl import app, flags
from bax import TrainState
from bax import Trainer
from bax.callbacks import CheckpointCallback, Callback
from chex import ArrayTree, Array
from ml_collections.config_flags import config_flags

from posterior_matching.models.vqvae import VQVAE
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
        self, reconstruction_fn: Callable[[ArrayTree], Array], dataset: tf.data.Dataset
    ):
        reconstruction_fn = hk.transform_with_state(reconstruction_fn).apply
        self._reconstruction_fn = jax.jit(reconstruction_fn)
        self._data_iter = dataset.unbatch().batch(3).repeat().as_numpy_iterator()
        self._prng = hk.PRNGSequence(random.randint(0, int(2e9)))

    def on_validation_end(
        self, train_state: TrainState, step: int, logs: Dict[str, Any]
    ):
        batch = next(self._data_iter)
        reconstructions, _ = self._reconstruction_fn(
            train_state.params, train_state.state, self._prng.next(), batch
        )

        x = np.broadcast_to(batch["image"], reconstructions.shape)
        reconstructions = np.concatenate([x, reconstructions], axis=2)

        assert np.all(np.logical_and(reconstructions <= 1.0, reconstructions >= 0.0))

        logs["reconstructions"] = reconstructions


def main(_):
    config = flags.FLAGS.config

    if "seed" not in config:
        config.seed = random.randint(0, int(2e9))

    config.lock()

    train_dataset, val_dataset = load_datasets(config.data)

    def loss_fn(step, is_training, batch):
        model = VQVAE(**config.model)
        out = model(batch["image"], is_training=is_training)
        aux = {
            "perplexity": jnp.mean(out["vq_output"]["perplexity"]),
            "reconstruction_loss": jnp.mean(out["reconstruction_loss"]),
            "vq_loss": jnp.mean(out["vq_output"]["loss"]),
        }
        return out["loss"], aux

    def reconstruction_fn(batch):
        model = VQVAE(**config.model)
        out = model(batch["image"], is_training=False)
        return jnp.clip(out["reconstruction"], 0.0, 1.0)

    optimizer = optax.adam(config.learning_rate)

    trainer = Trainer(loss_fn, optimizer, num_devices=1, seed=config.seed)

    run_dir = make_run_dir(prefix=f"vqvae-{config.data.dataset}")
    print("Using run directory:", run_dir)

    with open(os.path.join(run_dir, "model_config.json"), "w") as fp:
        json.dump(config.model.to_dict(), fp)

    callbacks = [
        CheckpointCallback(os.path.join(run_dir, "train_state.pkl")),
        ReconstructionCallback(reconstruction_fn, val_dataset),
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
