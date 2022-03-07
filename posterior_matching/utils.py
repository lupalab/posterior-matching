import os
import random
from datetime import datetime
from typing import Tuple, Mapping, Dict, Any, Optional, Callable

import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from bax import TrainState
from bax.callbacks import Callback
from chex import ArrayTree, Array
from optax import Schedule

from posterior_matching.masking import get_add_mask_fn, get_mask_generator


def configure_environment():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    tf.config.set_visible_devices([], "GPU")


def make_run_dir(path: str = "runs", prefix: Optional[str] = None) -> str:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if prefix is not None:
        run_id = prefix + "-" + run_id
    run_dir = os.path.join(path, run_id)
    os.makedirs(run_dir)
    return run_dir


def load_datasets(
    config: Mapping, normalize_images: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset = "mnist" if "mnist" in config["dataset"] else config["dataset"]
    train = tfds.load(dataset, split=config.get("train_split", "train"))
    val = tfds.load(dataset, split=config.get("validation_split", "validation"))

    train = train.shuffle(config.get("buffer_size", 40000))

    train = train.batch(config["train_batch_size"], drop_remainder=True)
    val = val.batch(config["val_batch_size"], drop_remainder=True)

    if "image" in train.element_spec:

        def rescale(x):
            x["image"] = tf.cast(x["image"], tf.float32)
            if normalize_images:
                x["image"] = x["image"] / 255.0
            return x

        train = train.map(rescale)
        val = val.map(rescale)

    if "id" in train.element_spec:

        def remove_id(x):
            del x["id"]
            return x

        train = train.map(remove_id)
        val = val.map(remove_id)

    if config["dataset"] == "celeb_a":

        def crop_and_resize(x):
            img = x["image"]
            img = img[:, 45:-45, 25:-25, :]
            img = tf.image.resize(img, (64, 64))
            return {"image": img}

        train = train.map(crop_and_resize)
        val = val.map(crop_and_resize)

    if "mnist16" in config["dataset"]:

        def resize(x):
            x["image"] = tf.image.resize(x["image"], (16, 16))
            return x

        train = train.map(resize).cache()
        val = val.map(resize).cache()

    if config["dataset"] == "mnist16_flat":

        def flatten(x):
            x["features"] = einops.rearrange(x["image"], "b ... -> b (...)")
            del x["image"]
            return x

        train = train.map(flatten)
        val = val.map(flatten)

    if "mask_generator" in config:
        mask_fn = get_add_mask_fn(
            get_mask_generator(
                config["mask_generator"], **config.get("mask_generator_kwargs", {})
            )
        )

        train = train.map(mask_fn)
        val = val.map(mask_fn)

    if "training_noise" in config:

        def add_noise(d):
            d["features"] += tf.random.normal(
                d["features"].shape, stddev=config.get("training_noise")
            )
            return d

        train = train.map(add_noise)

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)

    return train, val


def cyclical_annealing_schedule(
    low_value: float, high_value: float, period: int, delay: int = 0
) -> Schedule:
    def schedule(count):
        true_count = count
        count -= delay
        count = jnp.clip(count % period, 0, period // 2)
        frac = 1 - count / (period // 2)
        x = (low_value - high_value) * frac + high_value
        x *= true_count >= delay
        return x

    return schedule


class TensorBoardCallback(Callback):
    def __init__(self, path: str):
        self._writer = tf.summary.create_file_writer(path)

    def on_validation_end(
        self, train_state: TrainState, step: int, logs: Dict[str, Any]
    ):
        with self._writer.as_default():
            for k, v in logs.items():
                if np.ndim(v) != 0:
                    tf.summary.image(k, v, step=step)
                else:
                    tf.summary.scalar(k, v, step=step)


def batch_process(
    fn: Callable[[ArrayTree], Array],
    params: hk.Params,
    state: hk.State,
    dataset: tf.data.Dataset,
    seed: Optional[int] = None,
) -> Array:
    fn = jax.jit(hk.transform_with_state(fn).apply)
    prng = hk.PRNGSequence(seed or random.randint(0, int(2e9)))
    results = []
    for batch in dataset.as_numpy_iterator():
        results.append(fn(params, state, prng.next(), batch)[0])
    return jnp.concatenate(results, axis=0)
