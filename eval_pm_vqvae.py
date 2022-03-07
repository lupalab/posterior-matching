import json
import os
import pickle

import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import ray
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import app, flags
from tqdm import tqdm

from posterior_matching.masking import (
    get_add_mask_fn,
    get_mask_generator,
)
from posterior_matching.models.pixel_cnn import PixelCNN
from posterior_matching.models.vqvae import VQVAE, VQVAEPartialEncoder, vqvae_impute
from posterior_matching.prd.inception import get_inception_embeddings
from posterior_matching.prd.prd_score import (
    compute_prd_from_embedding,
    prd_to_max_f_beta_pair,
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
flags.DEFINE_string(
    "mask_generator",
    default=None,
    help="The name of the mask generator to use.",
    required=True,
)
flags.DEFINE_integer(
    "num_instances", default=None, help="The number of instances to evaluate."
)
flags.DEFINE_integer("batch_size", default=32, help="The batch size.")
flags.DEFINE_integer(
    "num_samples", default=10, help="The number of samples to use for expectations."
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

    with open(os.path.join(flags.FLAGS.run_dir, "vqvae_config.json"), "r") as fp:
        vqvae_config = json.load(fp)

    with open(os.path.join(flags.FLAGS.run_dir, "config.json"), "r") as fp:
        config = json.load(fp)

    with open(os.path.join(flags.FLAGS.run_dir, "train_state.pkl"), "rb") as fp:
        model_state = pickle.load(fp)

    def eval_fn(batch):
        vqvae = VQVAE(**vqvae_config)
        partial_encoder = VQVAEPartialEncoder(config["conditional_dim"], vqvae_config)
        partial_posterior = PixelCNN(**config["pixel_cnn"])

        imputations = vqvae_impute(
            vqvae,
            partial_encoder,
            partial_posterior,
            batch["image"],
            batch["mask"],
            num_samples=flags.FLAGS.num_samples,
        )
        mean_imputation = jnp.mean(imputations, axis=1)

        mse = jnp.mean((mean_imputation - batch["image"]) ** 2, axis=(1, 2, 3))
        psnr = -10.0 * jnp.log10(mse)

        return psnr, imputations

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

    psnrs = []
    prd_data = []
    total = dataset.cardinality().numpy().item()

    real_ds = dataset if num_devices < 2 else dataset.unbatch()
    real_images = np.concatenate(
        [x["image"] for x in real_ds.as_numpy_iterator()], axis=0
    )
    real_embeddings = get_inception_embeddings(real_images, batch_size=16)
    del real_images

    ray.init(num_cpus=flags.FLAGS.num_samples)

    for trial in range(flags.FLAGS.num_trials):
        imputations = []
        psnrs.append([])

        for batch in tqdm(
            dataset.as_numpy_iterator(), total=total, desc=f"Trial {trial + 1}"
        ):
            if num_devices > 1:
                key = jax.random.split(prng.next(), num_devices)
            else:
                key = prng.next()

            (psnr, imp), _ = eval_fn(params, state, key, batch)
            psnr = np.asarray(psnr)
            imp = np.asarray(imp)

            if num_devices > 1:
                psnr = einops.rearrange(psnr, "d b ... -> (d b) ...")
                imp = einops.rearrange(imp, "d b ... -> (d b) ...")

            psnrs[-1].append(psnr)
            imputations.append(imp)

        psnrs[-1] = np.concatenate(psnrs[-1], axis=0)
        imputations = np.concatenate(imputations, axis=0)

        @ray.remote
        def get_embeddings(x):
            return get_inception_embeddings(
                x, batch_size=16, verbose=False
            )

        fake_embeddings = ray.get(
            [
                get_embeddings.remote(imputations[:, i])
                for i in range(flags.FLAGS.num_samples)
            ]
        )

        fake_embeddings = np.concatenate([x[:, None] for x in fake_embeddings], axis=1)

        prd_data.append([])
        for i in tqdm(range(flags.FLAGS.num_samples), desc="Computing PRD"):
            prd_data[-1].append(
                compute_prd_from_embedding(
                    eval_data=fake_embeddings[:, i],
                    ref_data=real_embeddings,
                    num_clusters=20,
                    num_angles=1001,
                    num_runs=10,
                )
            )
        prd_data[-1] = np.array(prd_data[-1])

    psnrs = np.array(psnrs)
    prd_data = np.array(prd_data)

    per_trial_psnr = np.mean(np.ma.masked_invalid(psnrs), axis=1).data
    per_trial_prd = np.mean(prd_data, axis=1)

    f_scores = [prd_to_max_f_beta_pair(x[0], x[1], beta=8) for x in per_trial_prd]
    f_scores = np.array(f_scores)

    f_scores_means = np.mean(f_scores, axis=0)
    f_scores_stds = np.std(f_scores, axis=0)

    results_dir = os.path.join(flags.FLAGS.run_dir, "imputation_results")
    os.makedirs(results_dir, exist_ok=True)

    np.save(os.path.join(results_dir, "psnrs.npy"), psnrs)
    np.save(os.path.join(results_dir, "prd_data.npy"), prd_data)
    np.save(os.path.join(results_dir, "f_scores.npy"), f_scores)

    print("\n****RESULTS****")
    print(f"PSNR: {np.mean(per_trial_psnr).item()} ± {np.std(per_trial_psnr).item()}")
    print(f"Precision: {f_scores_means[1]} ± {f_scores_stds[1]}")
    print(f"Recall: {f_scores_means[0]} ± {f_scores_stds[0]}")


if __name__ == "__main__":
    app.run(main)
