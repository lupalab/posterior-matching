import functools
import json
import os
import pickle

import einops
import haiku as hk
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from tqdm import tqdm

from posterior_matching.acquisition import (
    make_acquisition_eval_fn,
    make_collect_trajectory_fn,
)
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
    "num_instances", default=1000, help="The number of instances to evaluate."
)
flags.DEFINE_integer(
    "num_samples", default=50, help="The number of samples to use for expectations."
)
flags.DEFINE_integer(
    "episode_length", default=31, help="The length of episodes to collect."
)


def load_data(dataset, num_instances):
    dataset_ = "mnist" if "mnist" in dataset else dataset
    ds = tfds.load(dataset_, split="test")

    if num_instances is not None:
        ds = ds.take(num_instances)

    ds = ds.batch(32)

    def rescale(x):
        x["image"] = tf.cast(x["image"], tf.float32) / 255.0
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

    if "mnist16" in dataset:

        def resize(x):
            x["image"] = tf.image.resize(x["image"], (16, 16))
            return x

        ds = ds.map(resize).cache()

    if dataset == "mnist16_flat":

        def flatten(x):
            x["features"] = einops.rearrange(x["image"], "b ... -> b (...)")
            del x["image"]
            return x

        ds = ds.map(flatten)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    is_image_data = "image" in ds.element_spec
    data_key = "image" if is_image_data else "features"
    data = np.vstack([x[data_key] for x in ds.as_numpy_iterator()])
    return data


def main(_):
    data = load_data(flags.FLAGS.dataset, flags.FLAGS.num_instances)

    with open(os.path.join(flags.FLAGS.run_dir, "lookahead_config.json"), "r") as fp:
        lookahead_config = json.load(fp)

    with open(os.path.join(flags.FLAGS.run_dir, "pm_vae_config.json"), "r") as fp:
        pm_vae_config = json.load(fp)

    with open(os.path.join(flags.FLAGS.run_dir, "train_state.pkl"), "rb") as fp:
        model_state = pickle.load(fp)

    eval_fn = make_acquisition_eval_fn(
        lookahead_config, pm_vae_config, flags.FLAGS.num_samples
    )
    collect_trajectory = make_collect_trajectory_fn(eval_fn, flags.FLAGS.episode_length)

    collect_trajectory = jax.jit(hk.transform_with_state(collect_trajectory).apply)
    collect_trajectory = functools.partial(
        collect_trajectory, model_state.params, model_state.state
    )

    sampling_trajectories = []
    lookahead_trajectories = []

    prng = hk.PRNGSequence(91)

    for x in tqdm(data, unit="episodes"):
        (sampling_traj, look_traj), _ = collect_trajectory(prng.next(), x)

        sampling_trajectories.append(sampling_traj)
        lookahead_trajectories.append(look_traj)

        for k, v in sampling_trajectories[-1].items():
            sampling_trajectories[-1][k] = np.asarray(v)

        for k, v in lookahead_trajectories[-1].items():
            lookahead_trajectories[-1][k] = np.asarray(v)

        sampling_trajectories[-1]["truth"] = x
        lookahead_trajectories[-1]["truth"] = x

    results_dir = os.path.join(flags.FLAGS.run_dir, "trajectories")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "sampling_trajectories.pkl"), "wb") as fp:
        pickle.dump(sampling_trajectories, fp)

    with open(os.path.join(results_dir, "lookahead_trajectories.pkl"), "wb") as fp:
        pickle.dump(lookahead_trajectories, fp)


if __name__ == "__main__":
    app.run(main)
