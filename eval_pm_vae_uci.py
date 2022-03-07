import json
import os
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from tqdm import tqdm

from posterior_matching.masking import get_add_mask_fn, BernoulliMaskGenerator
from posterior_matching.models.vae import PosteriorMatchingVAE
from posterior_matching.utils import configure_environment

configure_environment()

flags.DEFINE_string(
    "run_dir",
    default=None,
    help="The run directory of the model to evaluate.",
    required=True,
)
flags.DEFINE_string(
    "dataset",
    default=None,
    help="The dataset to evaluate on.",
    required=True,
)
flags.DEFINE_integer(
    "num_instances", default=None, help="The number of instances to evaluate."
)
flags.DEFINE_integer("batch_size", default=32, help="The batch size.")
flags.DEFINE_integer(
    "num_samples", default=512, help="The number of samples to use for expectations."
)
flags.DEFINE_integer(
    "num_trials",
    default=5,
    help="The number of trials to compute means and std. over.",
)


def load_dataset(dataset, batch_size, num_instances):
    ds = tfds.load(dataset, split="test")
    if num_instances is not None:
        ds = ds.take(num_instances)
    ds = ds.batch(batch_size, drop_remainder=True)

    mask_fn = get_add_mask_fn(BernoulliMaskGenerator())

    ds = ds.map(mask_fn)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def nrmse_score(
    imputations: np.ndarray, true_data: np.ndarray, observed_mask: np.ndarray
) -> np.ndarray:
    error = (imputations - true_data) ** 2
    mse = np.sum(error, axis=-2) / np.count_nonzero(1.0 - observed_mask, axis=-2)
    nrmse = np.sqrt(mse) / np.std(true_data, axis=-2)
    return np.mean(nrmse, axis=-1)


def main(_):
    dataset = load_dataset(
        flags.FLAGS.dataset, flags.FLAGS.batch_size, flags.FLAGS.num_instances
    )

    data_np = np.vstack([b["features"] for b in dataset.as_numpy_iterator()])

    with open(os.path.join(flags.FLAGS.run_dir, "model_config.json"), "r") as fp:
        model_config = json.load(fp)

    with open(os.path.join(flags.FLAGS.run_dir, "train_state.pkl"), "rb") as fp:
        model_state = pickle.load(fp)

    def eval_fn(batch):
        model = PosteriorMatchingVAE.from_config(model_config)

        x = batch["features"]
        b = batch["mask"]

        imputed = model.impute(x, b, num_samples=flags.FLAGS.num_samples)
        imputed = jnp.mean(imputed, axis=0)
        _, log_p_xu_given_xo = model.is_log_prob(
            x, b, num_samples=flags.FLAGS.num_samples
        )

        return imputed, log_p_xu_given_xo

    eval_fn = jax.jit(hk.transform_with_state(eval_fn).apply)

    imputations = []
    masks = []
    lls = []
    prng = hk.PRNGSequence(91)

    for i in range(flags.FLAGS.num_trials):
        imputations.append([])
        masks.append([])
        lls.append([])

        for batch in tqdm(
            dataset.as_numpy_iterator(),
            desc=f"Sampling (Trial {i + 1}/{flags.FLAGS.num_trials})",
        ):
            (im, ll), _ = eval_fn(
                model_state.params, model_state.state, prng.next(), batch
            )

            imputations[-1].append(im)
            masks[-1].append(batch["mask"])
            lls[-1].append(ll)

        imputations[-1] = np.vstack(imputations[-1])
        masks[-1] = np.vstack(masks[-1])
        lls[-1] = np.hstack(lls[-1])

    imputations = np.array(imputations)
    masks = np.array(masks)
    lls = np.array(lls)
    x = np.broadcast_to(data_np[None], [flags.FLAGS.num_trials, *data_np.shape])
    nrmse = nrmse_score(imputations, x, masks)
    lls = np.mean(lls, axis=1)

    results_dir = os.path.join(flags.FLAGS.run_dir, "uci_results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "nrmse.npy"), nrmse)
    np.save(os.path.join(results_dir, "ac_lls.npy"), lls)

    print("\n****RESULTS****")
    print(f"NRMSE: {np.mean(nrmse).item()} ± {np.std(nrmse).item()}")
    print(f"AC LL: {np.mean(lls).item()} ± {np.std(lls).item()}")


if __name__ == "__main__":
    app.run(main)
