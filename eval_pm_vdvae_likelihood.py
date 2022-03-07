import json
import math
import os
import pickle

import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from tqdm import tqdm

from posterior_matching.masking import get_add_mask_fn, get_mask_generator
from posterior_matching.models.vdvae import PosteriorMatchingVDVAE
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
flags.DEFINE_string(
    "mask_generator",
    default=None,
    help="The name of the mask generator to use.",
    required=True,
)
flags.DEFINE_integer(
    "num_instances", default=None, help="The number of instances to evaluate."
)
# This default (per-device) batch size of 625 was used to most efficiently evaluate the
# 10000 MNIST test instances on 8 TPUv3 cores. On other hardware, a smaller batch size
# may be required.
flags.DEFINE_integer("batch_size", default=625, help="The per-device batch size.")
flags.DEFINE_integer(
    "num_samples", default=10000, help="The number of samples to use for expectations."
)
flags.DEFINE_integer(
    "num_trials",
    default=5,
    help="The number of trials to compute means and std. over.",
)


def load_dataset(dataset, mask_generator, batch_size, num_instances):
    ds = tfds.load(dataset, split="test")

    if num_instances is not None:
        ds = ds.take(num_instances)

    ds = ds.batch(batch_size, drop_remainder=True)

    def rescale(x):
        x["image"] = tf.cast(x["image"], tf.float32)
        return x

    ds = ds.map(rescale)

    if "id" in ds.element_spec:

        def remove_id(x):
            del x["id"]
            return x

        ds = ds.map(remove_id)

    if dataset == "celeb_a":

        def crop_and_resize(x):
            img = x["image"]
            img = img[:, 45:-45, 25:-25, :]
            img = tf.image.resize(img, (64, 64))
            return {"image": img}

        ds = ds.map(crop_and_resize)

    mask_fn = get_add_mask_fn(get_mask_generator(mask_generator))
    ds = ds.map(mask_fn)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def main(_):
    dataset = load_dataset(
        flags.FLAGS.dataset,
        flags.FLAGS.mask_generator,
        flags.FLAGS.batch_size,
        flags.FLAGS.num_instances,
    )

    with open(os.path.join(flags.FLAGS.run_dir, "model_config.json"), "r") as fp:
        model_config = json.load(fp)

    with open(os.path.join(flags.FLAGS.run_dir, "train_state.pkl"), "rb") as fp:
        model_state = pickle.load(fp)

    def eval_fn(batch):
        model = PosteriorMatchingVDVAE(**model_config)

        px, pxu_xo = model.is_log_probs(
            batch["image"], batch["mask"], flags.FLAGS.num_samples
        )

        return px, pxu_xo

    eval_fn = jax.jit(hk.transform_with_state(eval_fn).apply)

    num_devices = jax.local_device_count()

    if num_devices > 1:
        eval_fn = jax.pmap(eval_fn)
        dataset = dataset.batch(num_devices, drop_remainder=True)

        @jax.pmap
        def get_params_state(_):
            return model_state.params, model_state.state

        params, state = get_params_state(jnp.arange(num_devices))
    else:
        params, state = model_state.params, model_state.state

    prng = hk.PRNGSequence(91)

    x_lls = []
    xo_lls = []
    total = dataset.cardinality().numpy().item()

    for trial in range(flags.FLAGS.num_trials):
        x_lls.append([])
        xo_lls.append([])

        for batch in tqdm(
            dataset.as_numpy_iterator(), total=total, desc=f"Trial {trial + 1}"
        ):
            if num_devices > 1:
                key = jax.random.split(prng.next(), num_devices)
            else:
                key = prng.next()

            (px, pxo), _ = eval_fn(params, state, key, batch)
            px = np.array(px)
            pxo = np.array(pxo)

            if num_devices > 1:
                px = einops.rearrange(px, "d b ... -> (d b) ...")
                pxo = einops.rearrange(pxo, "d b ... -> (d b) ...")

            x_lls[-1].append(px)
            xo_lls[-1].append(pxo)

        x_lls[-1] = np.concatenate(x_lls[-1], axis=0)
        xo_lls[-1] = np.concatenate(xo_lls[-1], axis=0)

    x_lls = np.array(x_lls)
    xo_lls = np.array(xo_lls)

    bpd = -x_lls / (math.prod(model_config["image_shape"]) * np.log(2))
    ac_lls = x_lls - xo_lls

    results_dir = os.path.join(flags.FLAGS.run_dir, "likelihood_results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "x_lls.npy"), x_lls)
    np.save(os.path.join(results_dir, "xo_lls.npy"), xo_lls)
    np.save(os.path.join(results_dir, "bpd.npy"), bpd)

    # Due to the very large number of samples being using for importance sampling
    # across the duration of evaluation, we (very infrequently) end up with non-finite
    # likelihoods. Naively including them in the mean across all instances will result
    # in a non-finite mean, which is why we mask them out here. In our evaluations in
    # the paper, we found 0 BPD values got masked out and only 40 out of 50,000 AC
    # likelihood values were masked out.
    bpd = np.ma.masked_array(
        bpd, mask=(~np.isfinite(bpd)) | (bpd > 1e10) | (bpd < -1e10)
    )
    ac_lls = np.ma.masked_array(
        ac_lls, mask=(~np.isfinite(ac_lls)) | (ac_lls > 1e10) | (ac_lls < -1e10)
    )

    per_trial_ac_lls = np.mean(ac_lls, axis=1)
    per_trial_bpd = np.mean(bpd, axis=1)

    print("\n****RESULTS****")
    print(f"BPD: {np.mean(per_trial_bpd).item()} ± {np.std(per_trial_bpd).item()}")
    print(
        f"AC LL: {np.mean(per_trial_ac_lls).item()} ± {np.std(per_trial_ac_lls).item()}"
    )


if __name__ == "__main__":
    app.run(main)
