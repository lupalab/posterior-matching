import math
from typing import Any, Dict, Callable, Tuple, Union

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array, ArrayTree, Scalar

from posterior_matching.models.lookahead import LookaheadPosterior


def rmse(true: Array, pred: Array, b: Array) -> Scalar:
    se = (true - pred) ** 2 * (1 - b)
    return jnp.sqrt(jnp.mean(se))


def make_acquisition_eval_fn(
    lookahead_config: Dict[str, Any], pm_vae_config: Dict[str, Any], num_samples: int
) -> Callable[[ArrayTree], Dict[str, Array]]:
    """Creates a model evaluation function for use when performing greedy acquisition.

    This method will return a function that evaluates expected information gains
    using both the sampling-based method and a "lookahead" model.

    Args:
        lookahead_config: Configuration for the lookahead model.
        pm_vae_config: Configuration for the Posterior Matching VAE model.
        num_samples: The number of samples to use when estimating information gains.

    Returns:
        A function that with the signature x_o, b -> data, where data is a dictionary
        containing information from the models. It is assumed that x_o and b will be
        single instances, i.e. not batched.
    """

    def eval_fn(x_o: Array, b: Array) -> Dict[str, Union[Array, Scalar]]:
        model = LookaheadPosterior.from_config(lookahead_config, pm_vae_config)

        sampling_info_gains = model.pm_vae.expected_info_gains(x_o, b, num_samples)
        lookahead_info_gains = model.expected_info_gains(x_o, b)

        pi_sampling = distrax.Categorical(
            jnp.where(sampling_info_gains == -jnp.inf, -1e10, sampling_info_gains)
        )
        pi_lookahead = distrax.Categorical(
            jnp.where(lookahead_info_gains == -jnp.inf, -1e10, lookahead_info_gains)
        )

        imputations = model.pm_vae.impute(
            jnp.expand_dims(x_o, 0), jnp.expand_dims(b, 0), num_samples
        )
        reconstruction = jnp.squeeze(jnp.mean(imputations, 0), 0)

        data = {
            "sampling_action": pi_sampling.mode(),
            "lookahead_action": pi_lookahead.mode(),
            "sampling_probs": pi_sampling.probs,
            "lookahead_probs": pi_lookahead.probs,
            "reconstruction": reconstruction,
        }

        return data

    return eval_fn


def make_collect_trajectory_fn(
    eval_fn: Callable[[ArrayTree], Dict[str, Array]], episode_length: int
) -> Callable[[Array], Tuple[Array, Array]]:
    """Creates a function that collects trajectories using the provided eval function.

    The returned function will execute active feature acquisition, using the actions
    that are defined by the provided `eval_fn`. The returned function is a pure Jax
    function, and therefore is very efficient.

    The acquisition procedure will be executed twice in parallel: one trajectory will
    use actions based on the sampling approach to information gain estimation, and the
    other trajectory will use actions based on the lookahead posteriors.

    Args:
        eval_fn: A function that was returned by `make_acquisition_eval_fn`.
        episode_length: The number of acquisitions to make.

    Returns:
        A function that accepts a single data instance and then simulates the
        active acquisition procedure for that instance, returning data about the two
        trajectories that were collected.
    """

    def collect_trajectory(x: Array) -> Tuple[Array, Array]:
        def step_with_sampling(cur_b, _):
            x_o = x * cur_b
            data = eval_fn(x_o, cur_b)
            num_features = math.prod(cur_b.shape)
            new_b = cur_b + jnp.reshape(
                jax.nn.one_hot(data["sampling_action"], num_features), cur_b.shape
            )

            data["rmse"] = rmse(x, data["reconstruction"], cur_b)
            data["mask"] = cur_b

            return new_b, data

        def step_with_lookahead(cur_b, _):
            x_o = x * cur_b
            data = eval_fn(x_o, cur_b)
            num_features = math.prod(cur_b.shape)
            new_b = cur_b + jnp.reshape(
                jax.nn.one_hot(data["lookahead_action"], num_features), cur_b.shape
            )

            data["rmse"] = rmse(x, data["reconstruction"], cur_b)
            data["mask"] = cur_b

            return new_b, data

        _, sampling_data = hk.scan(
            step_with_sampling, jnp.zeros_like(x), None, length=episode_length
        )
        _, lookahead_data = hk.scan(
            step_with_lookahead, jnp.zeros_like(x), None, length=episode_length
        )

        return sampling_data, lookahead_data

    return collect_trajectory
