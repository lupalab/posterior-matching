import math
from typing import Mapping, Any, Optional, Dict, Tuple

import einops
import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from chex import Array
from tensorflow_probability.substrates.jax.math import reduce_logmeanexp

from posterior_matching.models.distributions import get_distribution
from posterior_matching.models.networks import get_network


class PosteriorMatchingVAE(hk.Module):
    """A simple VAE, augmented with an extra encoder for doing Posterior Matching.

    Args:
        latent_dim: The dimensionality of the VAE's latent space.
        encoder_net: The encoder network to use.
        decoder_net: The decoder network to use.
        partial_encoder_net: The partial encoder network to use.
        posterior_dist: The distribution to be outputted by the encoder.
        decoder_dist: The distribution to be outputted by the decoder.
        partial_posterior_dist: The distribution to be outputted by the partial encoder.
        matching_ll_stop_gradients: If True, gradients will be stopped on samples
            from the posterior when computing the Posterior Matching loss. This means
            that the Posterior Matching loss will only propagate gradients to the
            parameters of the partial encoder.
        name: The optional name of the module.
    """

    def __init__(
        self,
        latent_dim: int,
        encoder_net: hk.Module,
        decoder_net: hk.Module,
        partial_encoder_net: hk.Module,
        posterior_dist: hk.Module,
        decoder_dist: hk.Module,
        partial_posterior_dist: hk.Module,
        matching_ll_stop_gradients: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.latent_dim = latent_dim
        self.encoder = hk.Sequential([encoder_net, posterior_dist], name="encoder")
        self.decoder = hk.Sequential([decoder_net, decoder_dist], name="decoder")
        self.partial_encoder = hk.Sequential(
            [partial_encoder_net, partial_posterior_dist], name="partial_encoder"
        )

        self.prior = tfd.MultivariateNormalDiag(
            jnp.zeros((latent_dim,)), jnp.ones((latent_dim,))
        )

        self._matching_ll_stop_gradients = matching_ll_stop_gradients

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], name: Optional[str] = None
    ) -> "PosteriorMatchingVAE":
        """Creates a `PosteriorMatchingVAE` model from a configuration dictionary.

        Args:
            config: The config for the model.
            name: The optional name of the module.

        Returns:
            A `PosteriorMatchingVAE` model.
        """
        encoder_net = get_network(
            config["encoder_net"], config.get("encoder_net_config"), name="encoder_net"
        )
        decoder_net = get_network(
            config["decoder_net"], config.get("decoder_net_config"), name="decoder_net"
        )
        partial_encoder_net = get_network(
            config.get("partial_encoder_net", config["encoder_net"]),
            config.get("partial_encoder_net_config", config.get("encoder_net_config")),
            name="partial_encoder_net",
        )

        posterior_dist_config = config.get("posterior_dist_config", {})
        posterior_dist_config["event_size"] = config["latent_dim"]
        partial_posterior_dist_config = config.get(
            "partial_posterior_dist_config", posterior_dist_config
        )
        partial_posterior_dist_config["event_size"] = config["latent_dim"]

        posterior_dist = get_distribution(
            config["posterior_dist"],
            posterior_dist_config,
            name="posterior_dist",
        )
        decoder_dist = get_distribution(
            config["decoder_dist"],
            config.get("decoder_dist_config"),
            name="decoder_dist",
        )
        partial_posterior_dist = get_distribution(
            config.get("partial_posterior_dist", config["posterior_dist"]),
            partial_posterior_dist_config,
            name="partial_posterior_dist",
        )
        return cls(
            config["latent_dim"],
            encoder_net,
            decoder_net,
            partial_encoder_net,
            posterior_dist,
            decoder_dist,
            partial_posterior_dist,
            config.get("matching_ll_stop_gradients", False),
            name=name,
        )

    def __call__(
        self, x: Array, b: Array, is_training: bool = False
    ) -> Dict[str, Array]:
        posterior = self.encoder(x, is_training=is_training)
        z = posterior.sample(seed=hk.next_rng_key())
        decoded = self.decoder(z, is_training=is_training)

        reconstruction_ll = decoded.log_prob(x)
        reconstruction_ll = einops.reduce(reconstruction_ll, "b ... -> b", "sum")

        kl = posterior.kl_divergence(self.prior)

        x_o = x * b
        x_o_b = jnp.concatenate([x_o, b], axis=-1)
        partial_posterior = self.partial_encoder(x_o_b, is_training=is_training)

        if self._matching_ll_stop_gradients:
            z = jax.lax.stop_gradient(z)
        matching_ll = partial_posterior.log_prob(z)

        return {
            "reconstruction_ll": reconstruction_ll,
            "kl": kl,
            "matching_ll": matching_ll,
        }

    def impute(self, x_o: Array, b: Array, num_samples: int = 100) -> Array:
        """Imputes unobserved features based on observed ones.

        Args:
            x_o: The observed feature values.
            b: The binary mask indicating which features are observed.
            num_samples: The number of imputations to generate per instance.

        Returns:
            A tensor of shape `[num_samples, *x_o.shape]`, which has imputed values
            at locations where `b == 0`.
        """
        x_o *= b
        x_o_b = jnp.concatenate([x_o, b], axis=-1)
        partial_posterior = self.partial_encoder(x_o_b)

        z = partial_posterior.sample(seed=hk.next_rng_key(), sample_shape=num_samples)
        x_u_samples = jax.vmap(lambda u: self.decoder(u).mean())(z)

        imputations = jnp.where(
            jnp.expand_dims(b, 0), jnp.expand_dims(x_o, 0), x_u_samples
        )

        return imputations

    def is_log_prob(
        self, x: Array, b: Array, num_samples: int = 100
    ) -> Tuple[Array, Array]:
        """Estimates marginal log probabilities using importance sampling.

        This function will compute both `log p(x)` and `log p(x_u | x_o)`.

        Args:
            x: The feature values.
            b: The binary mask indicating which features are observed.
            num_samples: The number of importance samples to use.

        Returns:
            Two arrays, the first of which contains `log p(x)` and the second of
            which contains `log p(x_u | x_o)`.
        """
        x_o = x * b
        x_o_b = jnp.concatenate([x_o, b], axis=-1)
        posterior = self.encoder(x)
        partial_posterior = self.partial_encoder(x_o_b)

        z = posterior.sample(seed=hk.next_rng_key(), sample_shape=num_samples)
        z_xo = partial_posterior.sample(
            seed=hk.next_rng_key(), sample_shape=num_samples
        )

        def decoder_ll(z):
            lls = self.decoder(z).log_prob(x)
            lls = einops.reduce(lls, "b ... -> b", "sum")
            return lls

        def posterior_ll(z):
            return posterior.log_prob(z)

        def decoder_ll_xo(z):
            lls = self.decoder(z).log_prob(x) * b
            lls = einops.reduce(lls, "b ... -> b", "sum")
            return lls

        def partial_posterior_ll(z):
            return partial_posterior.log_prob(z)

        log_p_z = self.prior.log_prob(z)
        log_p_z_xo = self.prior.log_prob(z_xo)

        log_p_xgz = jax.vmap(decoder_ll)(z)
        log_q_zgx = jax.vmap(posterior_ll)(z)

        log_p_xogz = jax.vmap(decoder_ll_xo)(z_xo)
        log_q_zgxo = jax.vmap(partial_posterior_ll)(z_xo)

        log_p_x = reduce_logmeanexp(log_p_xgz + log_p_z - log_q_zgx, axis=0)
        log_p_xo = reduce_logmeanexp(log_p_xogz + log_p_z_xo - log_q_zgxo, axis=0)
        log_p_xu_given_xo = log_p_x - log_p_xo

        return log_p_x, log_p_xu_given_xo

    def expected_info_gains(self, x: Array, b: Array, num_samples: int = 100) -> Array:
        """Computes the expected information gains when acquiring each feature.

        This function returns an array, with as many elements as there are features
        in the data, where each value in the array represents the expected information
        gain of `z` if that indices feature were to become observed. Indices that
        correspond to already observed features will have the value `-jnp.inf`.

        The expected information gains are computed using a sampling-based approach
        that can become somewhat expensive. See `lookahead.py` for a much more
        efficient method for estimating these information gains.

        Args:
            x: The feature values. Note that this is expected to be a single instance,
                i.e. there cannot be a batch dimension.
            b: The binary mask indicating which features are observed. As with `x`,
                this is expected to be a single instance, i.e. there cannot be a
                batch dimension.
            num_samples: The number of samples to use for estimating expectations.

        Returns:
            An array of shape `(num_features,)` where the ith element contains the
            expected information gain from acquiring feature i.
        """
        x_o = x * b
        x_o_b = jnp.concatenate([x_o, b], axis=-1)

        partial_posterior = self.partial_encoder(jnp.expand_dims(x_o_b, 0))
        z = partial_posterior.sample(seed=hk.next_rng_key(), sample_shape=num_samples)
        z = jnp.squeeze(z, 1)
        x_u_samples = self.decoder(z).mean()

        num_features = math.prod(b.shape)
        one_hots = jnp.eye(num_features)
        one_hots = jnp.reshape(one_hots, [num_features, *b.shape])

        batch_masks = jnp.maximum(jnp.expand_dims(b, 0), one_hots)
        batch_masks = jnp.concatenate([jnp.expand_dims(b, 0), batch_masks], axis=0)

        x_o_u_samples = jnp.where(
            jnp.expand_dims(b, 0) == 1,
            jnp.expand_dims(x_o, 0),
            x_u_samples,
        )

        def scan_func(c, x):
            x = jnp.broadcast_to(jnp.expand_dims(x, 0), batch_masks.shape)
            post = self.partial_encoder(
                jnp.concatenate([x * batch_masks, batch_masks], axis=-1)
            )
            return c, post.entropy()

        _, ents = hk.scan(scan_func, None, x_o_u_samples)
        ents = jnp.mean(ents, 0)

        ent_before = ents[0]
        ents_after = jnp.reshape(ents[1:], b.shape)

        info_gains = jnp.reshape(ent_before - ents_after, b.shape)
        info_gains = jnp.where(b == 0, info_gains, -jnp.inf)
        info_gains = jnp.reshape(info_gains, [-1])

        return info_gains
