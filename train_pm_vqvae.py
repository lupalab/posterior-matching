import json
import os
import pickle
import random
from typing import Callable, Dict, Any

import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from absl import app, flags
from bax import Trainer, TrainState
from bax.callbacks import CheckpointCallback, Callback
from chex import ArrayTree, Array
from ml_collections.config_flags import config_flags

from posterior_matching.models.pixel_cnn import PixelCNN
from posterior_matching.models.vqvae import VQVAE, VQVAEPartialEncoder, vqvae_impute
from posterior_matching.utils import (
    configure_environment,
    load_datasets,
    make_run_dir,
    TensorBoardCallback,
)

configure_environment()

config_flags.DEFINE_config_file("config", lock_config=False)


class ImputationCallback(Callback):
    def __init__(
        self,
        imputation_fn: Callable[[ArrayTree], Array],
        dataset: tf.data.Dataset,
    ):
        imputation_fn = hk.transform_with_state(imputation_fn).apply
        self._imputation_fn = jax.jit(imputation_fn)
        self._data_iter = dataset.unbatch().batch(3).repeat().as_numpy_iterator()
        self._prng = hk.PRNGSequence(random.randint(0, int(2e9)))

    def on_validation_end(
        self, train_state: TrainState, step: int, logs: Dict[str, Any]
    ):
        batch = next(self._data_iter)
        imputations, _ = self._imputation_fn(
            train_state.params, train_state.state, self._prng.next(), batch
        )

        assert np.all(np.logical_and(imputations <= 1.0, imputations >= 0.0))

        x = batch["image"]
        x_o = np.where(batch["mask"] == 1, x, 0.5)

        imputations = einops.rearrange(imputations, "b s h w c -> b h (s w) c")
        images = np.concatenate([x, x_o, imputations], axis=2)

        logs["imputations"] = images


def main(_):
    config = flags.FLAGS.config

    if "seed" not in config:
        config.seed = random.randint(0, int(2e9))

    train_dataset, val_dataset = load_datasets(config.data)

    with open(os.path.join(config.vqvae_dir, "model_config.json"), "r") as fp:
        vqvae_config = json.load(fp)

    with open(os.path.join(config.vqvae_dir, "train_state.pkl"), "rb") as fp:
        vqvae_state = pickle.load(fp)

    config.pixel_cnn.num_indices = vqvae_config["num_embeddings"]
    config.lock()

    def loss_fn(step, is_training, batch):
        vqvae = VQVAE(**vqvae_config)
        partial_encoder = VQVAEPartialEncoder(config.conditional_dim, vqvae_config)
        partial_posterior = PixelCNN(**config.pixel_cnn)

        encoding_indices = vqvae(batch["image"])["vq_output"]["encoding_indices"]
        x_o_b = jnp.concatenate(
            [batch["image"] * batch["mask"], batch["mask"]], axis=-1
        )
        cond_latents = partial_encoder(x_o_b)

        loss = -jnp.mean(
            partial_posterior.log_prob(
                encoding_indices,
                training=is_training,
                conditional_input=cond_latents,
            )
        )
        return loss, {}

    def imputation_fn(batch):
        vqvae = VQVAE(**vqvae_config)
        partial_encoder = VQVAEPartialEncoder(config.conditional_dim, vqvae_config)
        partial_posterior = PixelCNN(**config.pixel_cnn)

        return vqvae_impute(
            vqvae,
            partial_encoder,
            partial_posterior,
            batch["image"],
            batch["mask"],
            num_samples=5,
        )

    schedule = optax.exponential_decay(**config.lr_schedule)
    optimizer = optax.chain(
        optax.scale_by_adam(**config.get("adam", {})),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )

    def trainable_predicate(module_name, name, value):
        return not module_name.startswith("vqvae/")

    trainer = Trainer(
        loss_fn,
        optimizer,
        trainable_predicate=trainable_predicate,
        num_devices=1,
        seed=config.seed,
    )

    run_dir = make_run_dir(prefix=f"pm-vqvae-{config.data.dataset}")
    print("Using run directory:", run_dir)

    with open(os.path.join(run_dir, "config.json"), "w") as fp:
        json.dump(config.to_dict(), fp)

    with open(os.path.join(run_dir, "vqvae_config.json"), "w") as fp:
        json.dump(vqvae_config, fp)

    callbacks = [
        CheckpointCallback(os.path.join(run_dir, "train_state.pkl")),
        ImputationCallback(imputation_fn, val_dataset),
        TensorBoardCallback(os.path.join(run_dir, "tb")),
    ]

    trainer.fit(
        train_dataset,
        config.steps,
        val_dataset=val_dataset,
        validation_freq=config.validation_freq,
        callbacks=callbacks,
        initial_params=vqvae_state.params,
        initial_state=vqvae_state.state,
    )


if __name__ == "__main__":
    app.run(main)
