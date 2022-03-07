import json
import os
import pickle
import random

import jax.numpy as jnp
import optax
from absl import app, flags
from bax import Trainer
from bax.callbacks import CheckpointCallback, LearningRateLoggerCallback
from ml_collections.config_flags import config_flags

from posterior_matching.models.vade import PosteriorMatchingVADE
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

    config.data.mask_generator = "UniformMaskGenerator"
    config.lock()

    train_dataset, val_dataset = load_datasets(config.data)
    data_key = "image" if "image" in train_dataset.element_spec else "features"

    def loss_fn(step, is_training, batch):
        model = PosteriorMatchingVADE.from_config(config.model.to_dict())
        loss = -jnp.mean(model.posterior_matching_ll(batch[data_key], batch["mask"]))
        return loss, {}

    run_dir = make_run_dir(prefix=f"pm-vade-{config.data.dataset}")
    print("Using run directory:", run_dir)

    with open(os.path.join(config.vade_dir, "train_state.pkl"), "rb") as fp:
        vade_state = pickle.load(fp)

    schedule = optax.exponential_decay(**config.lr_schedule)
    optimizer = optax.chain(
        optax.scale_by_adam(**config.get("adam", {})),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )

    def trainable_predicate(module_name, name, value):
        return "partial_" in module_name

    trainer = Trainer(
        loss_fn,
        optimizer,
        num_devices=1,
        trainable_predicate=trainable_predicate,
        seed=config.seed,
    )

    callbacks = [
        CheckpointCallback(os.path.join(run_dir, "train_state.pkl")),
        LearningRateLoggerCallback(schedule),
        TensorBoardCallback(os.path.join(run_dir, "tb")),
    ]

    with open(os.path.join(run_dir, "model_config.json"), "w") as fp:
        json.dump(config.model.to_dict(), fp)

    print("Starting main training...")
    trainer.fit(
        train_dataset,
        config.steps,
        val_dataset=val_dataset,
        validation_freq=config.validation_freq,
        callbacks=callbacks,
        initial_params=vade_state.params,
        initial_state=vade_state.state,
    )


if __name__ == "__main__":
    app.run(main)
