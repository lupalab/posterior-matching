from typing import Mapping, Any

import distrax
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array
from tensorflow_probability.substrates.jax import distributions as tfd

from posterior_matching.models.distributions import (
    get_distribution,
    DiagonalGaussian,
    Independent,
)
from posterior_matching.models.networks import get_network


class VADE(hk.Module):
    """The Variational Deep Embedding model.

    Args:
        num_components: The number of mixture components in the prior.
        latent_dim: The dimensionality of the latent space.
        encoder_net: The encoder network to use.
        decoder_net: The decoder network to use.
        decoder_dist: The distribution to be outputted by the decoder.
    """

    def __init__(
        self,
        num_components: int,
        latent_dim: int,
        encoder_net: hk.Module,
        decoder_net: hk.Module,
        decoder_dist: hk.Module,
    ):
        super().__init__(name="vade")
        self.latent_dim = latent_dim

        logits = hk.get_parameter(
            name="logits",
            shape=(num_components,),
            init=hk.initializers.Constant(0),
        )
        loc = hk.get_parameter(
            name="mu",
            shape=(num_components, latent_dim),
            init=hk.initializers.RandomNormal(),
        )
        log_scale = hk.get_parameter(
            name="log_scale",
            shape=(num_components, latent_dim),
            init=hk.initializers.RandomNormal(),
        )
        self.pi = distrax.Categorical(logits)
        self.components = tfd.MultivariateNormalDiag(
            loc=loc, scale_diag=jnp.exp(log_scale)
        )

        self.encoder = hk.Sequential(
            [encoder_net, DiagonalGaussian(latent_dim)], name="encoder"
        )
        self.decoder = hk.Sequential(
            [decoder_net, decoder_dist, Independent()], name="decoder"
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "VADE":
        """Creates a `VADE` model from a configuration dictionary.

        Args:
            config: The config for the model.

        Returns:
            A `VADE` model.
        """
        encoder_net = get_network(
            config["encoder_net"], config.get("encoder_net_config"), name="encoder_net"
        )
        decoder_net = get_network(
            config["decoder_net"], config.get("decoder_net_config"), name="decoder_net"
        )
        decoder_dist = get_distribution(
            config["decoder_dist"],
            config.get("decoder_dist_config"),
            name="decoder_dist",
        )
        return cls(
            config["num_components"],
            config["latent_dim"],
            encoder_net,
            decoder_net,
            decoder_dist,
        )

    def predict_cluster(self, x: Array, num_samples: int = 10) -> Array:
        """Predicts the cluster assignment of some data.

        Args:
            x: The data to get cluster assignments for.
            num_samples: The number of samples to use when estimating the clusters.

        Returns:
            The probabilities over clusters for each data point.
        """
        posterior = self.encoder(x)
        z = posterior.sample(sample_shape=num_samples, seed=hk.next_rng_key())

        log_p_z_given_c = jax.vmap(jax.vmap(self.components.log_prob))(z)
        h = log_p_z_given_c + jnp.expand_dims(self.pi.logits, (0, 1))
        q_c_given_x = jnp.mean(jax.nn.softmax(h, axis=-1), 0)
        return q_c_given_x

    def elbo(self, x: Array) -> Array:
        """Computes the VaDE evidence lower bound.

        Args:
            x: The data to compute the ELBO for.

        Returns:
            The ELBO of `x`.
        """
        posterior = self.encoder(x)
        z = posterior.sample(seed=hk.next_rng_key())
        reconstruction = self.decoder(z)

        log_p_x_given_z = reconstruction.log_prob(x)
        log_p_z_given_c = jax.vmap(self.components.log_prob)(z)
        unnorm_log_q_c_given_x = log_p_z_given_c + jnp.expand_dims(self.pi.logits, 0)

        log_q_c_given_x = jax.nn.log_softmax(unnorm_log_q_c_given_x, axis=-1)
        log_q_z_given_x = posterior.log_prob(z)

        gamma = jnp.exp(log_q_c_given_x)
        e_log_p_z_given_c = jnp.einsum("bc,bc->b", log_p_z_given_c, gamma)
        e_log_p_c = jnp.einsum("c,bc->b", self.pi.logits, gamma)
        e_log_q_c_given_x = jnp.einsum("bc,bc->b", log_q_c_given_x, gamma)

        elbo = (
            log_p_x_given_z
            + e_log_p_z_given_c
            + e_log_p_c
            - log_q_z_given_x
            - e_log_q_c_given_x
        )

        return elbo


class PosteriorMatchingVADE(VADE):
    """A VADE model with an addition Posterior Matching encoder.

    Args:
        num_components: The number of mixture components in the prior.
        latent_dim: The dimensionality of the latent space.
        encoder_net: The encoder network to use.
        partial_encoder_net: The partial encoder network to use.
        partial_posterior_dist: The distribution to be outputted by the partial encoder.
        decoder_net: The decoder network to use.
        decoder_dist: The distribution to be outputted by the decoder.
    """

    def __init__(
        self,
        num_components: int,
        latent_dim: int,
        encoder_net: hk.Module,
        partial_encoder_net: hk.Module,
        partial_posterior_dist: hk.Module,
        decoder_net: hk.Module,
        decoder_dist: hk.Module,
    ):
        super().__init__(
            num_components, latent_dim, encoder_net, decoder_net, decoder_dist
        )

        self.partial_encoder = hk.Sequential(
            [partial_encoder_net, partial_posterior_dist], name="partial_encoder"
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PosteriorMatchingVADE":
        """Creates a `PosteriorMatchingVADE` model from a configuration dictionary.

        Args:
            config: The config for the model.

        Returns:
            A `PosteriorMatchingVADE` model.
        """
        encoder_net = get_network(
            config["encoder_net"], config.get("encoder_net_config"), name="encoder_net"
        )
        partial_encoder_net = get_network(
            config.get("partial_encoder_net", config["encoder_net"]),
            config.get("partial_encoder_net_config", config.get("encoder_net_config")),
            name="partial_encoder_net",
        )
        partial_posterior_dist_config = config.get("partial_posterior_dist_config")
        partial_posterior_dist_config["event_size"] = config["latent_dim"]
        partial_posterior_dist = get_distribution(
            config.get("partial_posterior_dist", "TriLGaussian"),
            partial_posterior_dist_config,
            name="partial_posterior_dist",
        )
        decoder_net = get_network(
            config["decoder_net"], config.get("decoder_net_config"), name="decoder_net"
        )
        decoder_dist = get_distribution(
            config["decoder_dist"],
            config.get("decoder_dist_config"),
            name="decoder_dist",
        )
        return cls(
            config["num_components"],
            config["latent_dim"],
            encoder_net,
            partial_encoder_net,
            partial_posterior_dist,
            decoder_net,
            decoder_dist,
        )

    def partial_predict_cluster(
        self, x: Array, b: Array, num_samples: int = 10
    ) -> Array:
        """Predicts the cluster assignment of some partially observed data.

        Args:
            x: The observed data values to get cluster assignments for.
            b: The binary mask indicating which features are observed.
            num_samples: The number of samples to use when estimating the clusters.

        Returns:
            The probabilities over clusters for each data point.
        """
        x_o_b = jnp.concatenate([x * b, b], axis=-1)
        partial_posterior = self.partial_encoder(x_o_b)
        z = partial_posterior.sample(sample_shape=num_samples, seed=hk.next_rng_key())

        log_p_z_given_c = jax.vmap(jax.vmap(self.components.log_prob))(z)
        h = log_p_z_given_c + jnp.expand_dims(self.pi.logits, (0, 1))
        q_c_given_x_o = jnp.mean(jax.nn.softmax(h, axis=-1), 0)
        return q_c_given_x_o

    def posterior_matching_ll(self, x: Array, b: Array) -> Array:
        """Compute the Posterior Matching loss.

        Args:
            x: The observed data values to get cluster assignments for.
            b: The binary mask indicating which features are observed.

        Returns:
            The Posterior Matching loss.
        """
        x_o = x * b
        x_o_b = jnp.concatenate([x_o, b], axis=-1)

        posterior = self.encoder(x)
        partial_posterior = self.partial_encoder(x_o_b)

        z = posterior.sample(seed=hk.next_rng_key())
        log_p_z_given_xo = partial_posterior.log_prob(jax.lax.stop_gradient(z))

        return log_p_z_given_xo
