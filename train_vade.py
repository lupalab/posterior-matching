import json
import os
import pickle
import random

import haiku as hk
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags
from bax import Trainer
from bax.callbacks import CheckpointCallback, LearningRateLoggerCallback
from ml_collections.config_flags import config_flags
from sklearn.mixture import GaussianMixture

from posterior_matching.clustering import (
    clustering_accuracy,
    ClusteringAccuracyCallback,
)
from posterior_matching.models.vade import VADE
from posterior_matching.utils import (
    configure_environment,
    load_datasets,
    make_run_dir,
    TensorBoardCallback,
    batch_process,
)

configure_environment()

config_flags.DEFINE_config_file("config", lock_config=False)


def main(_):
    config = flags.FLAGS.config

    if "seed" not in config:
        config.seed = random.randint(0, int(2e9))

    config.lock()

    train_dataset, val_dataset = load_datasets(config.data)
    data_key = "image" if "image" in train_dataset.element_spec else "features"

    def pretrain_loss_fn(step, is_training, batch):
        model = VADE.from_config(config.model)
        z = model.encoder(batch[data_key]).mean()
        loss = -jnp.mean(model.decoder(z).log_prob(batch[data_key]))
        return loss, {}

    def loss_fn(step, is_training, batch):
        model = VADE.from_config(config.model)
        elbo = model.elbo(batch[data_key])
        loss = -jnp.mean(elbo)
        return loss, {}

    def pred_fn(batch):
        model = VADE.from_config(config.model)
        probs = model.predict_cluster(batch[data_key], config.cluster_pred_num_samples)
        preds = jnp.argmax(probs, axis=-1)
        return preds

    def encode_fn(batch):
        model = VADE.from_config(config.model)
        return model.encoder(batch[data_key]).mean()

    run_dir = make_run_dir(prefix=f"vade-{config.data.dataset}")
    print("Using run directory:", run_dir)

    # PRETRAINING

    pretrain_trainer = Trainer(
        pretrain_loss_fn, optax.adam(config.pretrain_lr), seed=config.seed
    )

    print("Pretraining...")
    pretrain_state = pretrain_trainer.fit(train_dataset, config.pretrain_steps)

    with open(os.path.join(run_dir, "pretrain_state.pkl"), "wb") as fp:
        pickle.dump(pretrain_state, fp)

    # GMM

    print("Fitting GMM...")
    latents = batch_process(
        encode_fn,
        pretrain_state.params,
        pretrain_state.state,
        train_dataset,
        config.seed,
    )
    val_latents = batch_process(
        encode_fn,
        pretrain_state.params,
        pretrain_state.state,
        val_dataset,
        config.seed,
    )

    gmm = GaussianMixture(
        n_components=config.model.num_components,
        covariance_type="diag",
        max_iter=300,
        n_init=10,
    )

    gmm.fit(latents)
    gmm_preds = gmm.predict(val_latents)
    targets = np.concatenate(
        [x for x in val_dataset.map(lambda x: x["label"]).as_numpy_iterator()], axis=0
    )
    gmm_acc = clustering_accuracy(targets, gmm_preds)
    print("GMM Accuracy:", round(gmm_acc, 4))

    gmm_params = {
        "vade": {
            "logits": np.log(gmm.weights_),
            "mu": gmm.means_,
            "log_scale": np.log(gmm.covariances_),
        }
    }

    initial_params = hk.data_structures.merge(pretrain_state.params, gmm_params)

    # MAIN TRAINING

    with open(os.path.join(run_dir, "model_config.json"), "w") as fp:
        json.dump(config.model.to_dict(), fp)

    schedule = optax.exponential_decay(**config.lr_schedule)
    optimizer = optax.chain(
        optax.scale_by_adam(**config.get("adam", {})),
        optax.scale_by_schedule(schedule),
        optax.scale(-1.0),
    )

    trainer = Trainer(
        loss_fn,
        optimizer,
        num_devices=1,
        seed=config.seed,
    )

    callbacks = [
        ClusteringAccuracyCallback(pred_fn),
        CheckpointCallback(os.path.join(run_dir, "train_state.pkl")),
        LearningRateLoggerCallback(schedule),
        TensorBoardCallback(os.path.join(run_dir, "tb")),
    ]

    print("Starting main training...")
    trainer.fit(
        train_dataset,
        config.steps,
        val_dataset=val_dataset,
        validation_freq=config.validation_freq,
        callbacks=callbacks,
        initial_params=initial_params,
    )


if __name__ == "__main__":
    app.run(main)
