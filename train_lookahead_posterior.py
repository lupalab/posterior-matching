import json
import math
import os
import pickle
import random

import jax.numpy as jnp
import optax
from absl import app, flags
from bax import Trainer
from bax.callbacks import CheckpointCallback, LearningRateLoggerCallback
from ml_collections.config_flags import config_flags

from posterior_matching.models.lookahead import LookaheadPosterior
from posterior_matching.utils import (
    configure_environment,
    load_datasets,
    make_run_dir,
    TensorBoardCallback,
)

configure_environment()

config_flags.DEFINE_config_file("config", lock_config=False)


def main(_):
    config = flags.FLAGS.config

    if "seed" not in config:
        config.seed = random.randint(0, int(2e9))

    train_dataset, val_dataset = load_datasets(config.data)

    is_image_data = "image" in train_dataset.element_spec
    data_key = "image" if is_image_data else "features"

    with open(os.path.join(config.pm_vae_dir, "model_config.json"), "r") as fp:
        pm_vae_config = json.load(fp)

    with open(os.path.join(config.pm_vae_dir, "train_state.pkl"), "rb") as fp:
        pm_vae_state = pickle.load(fp)

    config.model.num_features = math.prod(train_dataset.element_spec["mask"].shape[1:])
    config.lock()

    def loss_fn(step, is_training, batch):
        model = LookaheadPosterior.from_config(config.model, pm_vae_config)
        lookahead_lls = model(batch[data_key], batch["mask"])

        loss = -jnp.mean(lookahead_lls)

        return loss, {}

    schedule = optax.exponential_decay(**config.lr_schedule)
    optimizer = optax.chain(
        optax.scale_by_adam(**config.get("adam", {})),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )

    def trainable_predicate(module_name, name, value):
        return "lookahead" in module_name

    trainer = Trainer(
        loss_fn,
        optimizer,
        trainable_predicate=trainable_predicate,
        num_devices=1,
        seed=config.seed,
    )

    run_dir = make_run_dir(prefix=f"lookahead-{config.data.dataset}")
    print("Using run directory:", run_dir)

    with open(os.path.join(run_dir, "lookahead_config.json"), "w") as fp:
        json.dump(config.model.to_dict(), fp)

    with open(os.path.join(run_dir, "pm_vae_config.json"), "w") as fp:
        json.dump(pm_vae_config, fp)

    callbacks = [
        CheckpointCallback(os.path.join(run_dir, "train_state.pkl")),
        LearningRateLoggerCallback(schedule),
        TensorBoardCallback(os.path.join(run_dir, "tb")),
    ]

    trainer.fit(
        train_dataset,
        config.steps,
        val_dataset=val_dataset,
        validation_freq=config.validation_freq,
        callbacks=callbacks,
        initial_params=pm_vae_state.params,
        initial_state=pm_vae_state.state,
    )


if __name__ == "__main__":
    app.run(main)
