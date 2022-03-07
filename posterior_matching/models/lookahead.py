from typing import Optional, Mapping, Any

import einops
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array
from tensorflow_probability.substrates.jax import distributions as tfd

from posterior_matching.models.networks import get_network
from posterior_matching.models.vae import PosteriorMatchingVAE


class LookaheadBlock(hk.Module):
    """This layer outputs one "lookahead" posterior for each feature."""

    def __init__(self, event_size, num_features, w_init=None, b_init=None, name=None):
        super().__init__(name=name)
        self._event_size = event_size
        self._num_features = num_features
        self._num_params = event_size * 2
        self._w_init = w_init
        self._b_init = b_init

    def __call__(self, inputs):
        x = einops.rearrange(inputs, "b ... -> b (...)")

        params = hk.Linear(
            self._num_params * self._num_features,
            w_init=self._w_init,
            b_init=self._b_init,
        )(x)

        params = einops.rearrange(
            params, "b (f p) -> b f p", f=self._num_features, p=self._num_params
        )

        loc = params[..., : self._event_size]
        scale = jax.nn.softplus(params[..., self._event_size :]) + 1e-5

        return tfd.MultivariateNormalDiag(loc, scale)


class LookaheadPosterior(hk.Module):
    """A model for learning lookahead posteriors for active feature acquisition.

    This class can be used to augment a `PosteriorMatchingVAE` with an additional
    encoder network that learns to output lookahead posteriors.

    Args:
        pm_vae: The `PosteriorMatchingVAE` that we want to learn lookahead posteriors
            for.
        lookahead_encoder_net: The encoder network to use for outputting lookahead
            posteriors.
        num_features: The number of features in the data, i.e. the number of lookahead
            posteriors that will be outputted.
        lookahead_subsample: The number of lookahead posteriors that will be randomly
            selected to be updated on each training step.
        model_samples: The number of samples from the `pm_vae` model to use for
            training the lookahead posteriors.
        name: The optional name of the module.
    """

    def __init__(
        self,
        pm_vae: PosteriorMatchingVAE,
        lookahead_encoder_net: hk.Module,
        num_features: int,
        lookahead_subsample: int = 16,
        model_samples: int = 64,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.pm_vae = pm_vae
        self.lookahead_encoder = hk.Sequential(
            [lookahead_encoder_net, LookaheadBlock(pm_vae.latent_dim, num_features)],
            name="lookahead_encoder",
        )

        self._num_features = num_features
        self._lookahead_subsample = lookahead_subsample
        self._model_samples = model_samples

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        pm_vae_config: Mapping[str, Any],
        name: Optional[str] = None,
    ):
        """Creates a `LookaheadPosterior` from configuration dictionaries.

        Args:
            config: The config for the lookahead posterior components of the model.
            pm_vae_config: The config for the `PosteriorMatchingVAE`. This will
                simply be forwarded to `PosteriorMatchingVAE.from_config`.
            name: The optional name of the module.

        Returns:
            A `LookaheadPosterior` model.
        """
        pm_vae = PosteriorMatchingVAE.from_config(pm_vae_config)

        lookahead_encoder_net = get_network(
            config.get("lookahead_encoder_net", pm_vae_config["encoder_net"]),
            config.get(
                "lookahead_encoder_net_config", pm_vae_config.get("encoder_net_config")
            ),
            name="lookahead_encoder_net",
        )

        return cls(
            pm_vae,
            lookahead_encoder_net,
            config["num_features"],
            config.get("lookahead_subsample", 16),
            config.get("model_samples", 64),
            name=name,
        )

    def __call__(self, x: Array, b: Array, is_training: bool = False) -> Array:
        x_o = x * b
        x_o_b = jnp.concatenate([x_o, b], axis=-1)

        po_posterior = self.pm_vae.partial_encoder(x_o_b, is_training=False)

        z = po_posterior.sample(
            seed=hk.next_rng_key(), sample_shape=self._model_samples
        )

        def decode(z):
            return self.pm_vae.decoder(z).mean()

        x_o_u_samples_look = jax.vmap(decode)(z)
        x_o_u_samples_look = jnp.where(
            jnp.expand_dims(b == 1, 0), jnp.expand_dims(x_o, 0), x_o_u_samples_look
        )

        one_hots = jnp.eye(self._num_features)
        one_hots = jnp.reshape(one_hots, [self._num_features, *b.shape[1:]])

        subsampled_inds = jax.random.choice(
            hk.next_rng_key(),
            self._num_features,
            (self._lookahead_subsample,),
            replace=False,
        )

        subsampled_one_hots = one_hots[subsampled_inds]

        b_look = jnp.maximum(
            jnp.expand_dims(b, 1), jnp.expand_dims(subsampled_one_hots, 0)
        )
        x_o_model_look = jnp.expand_dims(x_o_u_samples_look, 2) * b_look
        x_o_model_look = jax.lax.stop_gradient(x_o_model_look)

        valid_mask = (
            einops.reduce(
                jnp.expand_dims(b, 1) + jnp.expand_dims(subsampled_one_hots, 0),
                "b s ... -> b s",
                "max",
            )
            < 2
        )

        b_look = einops.rearrange(b_look, "b s ... -> (b s) ...")
        x_o_model_look = einops.rearrange(x_o_model_look, "z b s ... -> z (b s) ...")

        def model_sample(key, x_o):
            return self.pm_vae.partial_encoder(
                jnp.concatenate([x_o, b_look], axis=-1)
            ).sample(seed=key)

        model_one_step_z = jax.vmap(model_sample)(
            jax.random.split(hk.next_rng_key(), self._model_samples), x_o_model_look
        )

        model_one_step_z = einops.rearrange(
            model_one_step_z,
            "z (b s) ... -> z b s ...",
            z=self._model_samples,
            s=self._lookahead_subsample,
        )

        lookahead_posteriors = self.lookahead_encoder(x_o_b)
        lookahead_posteriors = lookahead_posteriors[:, subsampled_inds]

        model_lookahead_lls = jax.vmap(lookahead_posteriors.log_prob)(
            jax.lax.stop_gradient(model_one_step_z)
        )
        model_lookahead_lls = einops.rearrange(
            model_lookahead_lls, "(z b) ... -> z b ...", z=self._model_samples
        )

        lookahead_lls = jnp.mean(model_lookahead_lls, axis=0) * valid_mask

        denom = jnp.count_nonzero(valid_mask, axis=-1)
        lookahead_lls = jnp.sum(lookahead_lls, axis=-1) / denom
        lookahead_lls = jnp.where(denom == 0, 0, lookahead_lls)

        return lookahead_lls

    def expected_info_gains(self, x: Array, b: Array) -> Array:
        """Estimates the info gain for acquiring each feature, using the lookahead posteriors.

        Args:
            x: The feature values.
            b: The binary mask indicating which features are observed.

        Returns:
            An array containing the estimated information gain for each feature.
            The info gain for features that have already been observed will be
            set to `-jnp.inf`.
        """
        x_o = x * b
        x_o_b = jnp.concatenate([x_o, b], axis=-1)

        current_ent = self.pm_vae.encoder(jnp.expand_dims(x, 0)).entropy()

        lookahead_posteriors = self.lookahead_encoder(jnp.expand_dims(x_o_b, 0))
        lookahead_ents = lookahead_posteriors.entropy()

        info_gains = jnp.reshape(current_ent - lookahead_ents, b.shape)
        info_gains = jnp.where(b == 0, info_gains, -jnp.inf)
        info_gains = jnp.reshape(info_gains, [-1])
        return info_gains
