from typing import Optional, Mapping, Any

import distrax
import einops
import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from chex import Array, PRNGKey

from posterior_matching.models.networks import ResidualMLP


class Independent(hk.Module):
    def __call__(self, dist: tfd.Distribution) -> tfd.Distribution:
        return tfd.Independent(dist)


class Bernoulli(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def __call__(self, x: Array) -> tfd.Bernoulli:
        return tfd.Bernoulli(x)


class IdentityGaussian(hk.Module):
    def __init__(
        self,
        event_size: int,
        w_init: hk.initializers.Initializer = None,
        b_init: hk.initializers.Initializer = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._event_size = event_size
        self._w_init = w_init
        self._b_init = b_init

    def __call__(self, x: Array) -> tfd.Normal:
        x = hk.Flatten()(x)

        loc = hk.Linear(
            self._event_size,
            w_init=self._w_init,
            b_init=self._b_init,
        )(x)

        log_scale = hk.get_parameter(
            "log_scale", shape=(), init=hk.initializers.Constant(0.0)
        )
        scale = jnp.broadcast_to(jnp.exp(log_scale), loc.shape)

        return tfd.Normal(loc, scale)


class DiagonalGaussian(hk.Module):
    def __init__(
        self,
        event_size: int,
        w_init: hk.initializers.Initializer = None,
        b_init: hk.initializers.Initializer = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._event_size = event_size
        self._num_params = event_size * 2
        self._w_init = w_init
        self._b_init = b_init

    def __call__(self, x: Array) -> tfd.MultivariateNormalDiag:
        x = hk.Flatten()(x)

        params = hk.Linear(
            self._num_params,
            w_init=self._w_init,
            b_init=self._b_init,
        )(x)

        loc = params[:, : self._event_size]
        scale = jax.nn.softplus(params[:, self._event_size :]) + 1e-5

        return tfd.MultivariateNormalDiag(loc, scale)


class TriLGaussian(hk.Module):
    def __init__(
        self,
        event_size: int,
        w_init: hk.initializers.Initializer = None,
        b_init: hk.initializers.Initializer = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._event_size = event_size
        self._num_params = event_size + event_size * (event_size + 1) // 2
        self._w_init = w_init
        self._b_init = b_init

    def __call__(self, x: Array) -> tfd.MultivariateNormalTriL:
        x = hk.Flatten()(x)

        params = hk.Linear(
            self._num_params,
            w_init=self._w_init,
            b_init=self._b_init,
        )(x)

        loc = params[:, : self._event_size]
        scale = tfb.FillScaleTriL()(params[:, self._event_size :])

        return tfd.MultivariateNormalTriL(loc, scale)


class OneDimensionalGMM(hk.Module):
    def __init__(
        self, event_size: int, num_components: int = 10, name: Optional[str] = None
    ):
        super().__init__(name=name)
        self._event_size = event_size
        self._num_components = num_components

    def __call__(self, x: Array) -> tfd.MixtureSameFamily:
        params = hk.Linear(3 * self._num_components * self._event_size)(x)
        params = einops.rearrange(params, "b (d p) -> b d p", d=self._event_size)

        logits = params[..., : self._num_components]
        means = params[..., self._num_components : -self._num_components]
        scales = jax.nn.softplus(params[..., -self._num_components :]) + 1e-5

        components_dist = tfd.Normal(means, scales)
        mixture_dist = tfd.Categorical(logits)
        return tfd.MixtureSameFamily(mixture_dist, components_dist)


class _AutoregressiveDistribution(distrax.Distribution):
    def __init__(self, event_size: int, context: Array, net: hk.Module):
        self._event_size = event_size
        self._context = context
        self._net = net

    @property
    def event_shape(self):
        return (self._event_size,)

    def __getitem__(self, i):
        return _AutoregressiveDistribution(
            self._event_size, self._context[i : i + 1], self._net
        )

    def log_prob(self, value):
        def body_fun(c, i):
            mask = jnp.less(
                jnp.broadcast_to(
                    jnp.arange(self._event_size, dtype=value.dtype), value.shape
                ),
                i,
            )
            x_o = value * mask
            out = self._net(jnp.concatenate([x_o, mask, self._context], axis=-1))
            lls = out.log_prob(value)[:, i]
            return c, lls

        _, out = hk.scan(body_fun, None, jnp.arange(self._event_size))
        return jnp.sum(out, 0)

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        def sample_one(cond):
            cond = jnp.broadcast_to(jnp.expand_dims(cond, 0), [n, cond.shape[-1]])

            def body_fun(i, x):
                mask = jnp.less(
                    jnp.broadcast_to(
                        jnp.arange(self._event_size, dtype=x.dtype), x.shape
                    ),
                    i,
                )
                x_o = x.astype(self._context.dtype) * mask
                out = self._net(jnp.concatenate([x_o, mask, cond], axis=-1))
                b = jnp.broadcast_to(jnp.arange(x.shape[-1]) == i, x.shape)
                updates = out.sample(seed=key) * b
                return x + updates

            init_value = jnp.zeros((n, self._event_size), self._context.dtype)
            return hk.fori_loop(0, self._event_size, body_fun, init_value)

        samples = jax.vmap(sample_one)(self._context)
        return jnp.transpose(samples, [1, 0, 2])


class AutoregressiveGMM(hk.Module):
    """An autoregressive distribution that uses a GMM at each step.

    This is a relatively simple and naive incarnation of an autoregressive distribution.
    However, it tends to perform pretty well.

    See the Appendix of the paper for a description of this distribution.
    """

    def __init__(
        self,
        event_size: int,
        num_components: int = 10,
        residual_blocks: int = 2,
        hidden_units: int = 256,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._event_dim = event_size
        self._num_components = num_components
        self._residual_blocks = residual_blocks
        self._hidden_units = hidden_units

    def __call__(self, x) -> tfd.Distribution:
        net = hk.Sequential(
            [
                ResidualMLP(self._residual_blocks, self._hidden_units),
                OneDimensionalGMM(self._event_dim, self._num_components),
            ]
        )
        x = hk.Flatten()(x)
        return _AutoregressiveDistribution(self._event_dim, x, net)


_DISTRIBUTIONS = {
    "Bernoulli": Bernoulli,
    "IdentityGaussian": IdentityGaussian,
    "DiagonalGaussian": DiagonalGaussian,
    "TriLGaussian": TriLGaussian,
    "AutoregressiveGMM": AutoregressiveGMM,
}


def get_distribution(
    distribution_type: str,
    distribution_config: Optional[Mapping[str, Any]] = None,
    name: Optional[str] = None,
) -> hk.Module:
    distribution_config = distribution_config or {}
    return _DISTRIBUTIONS[distribution_type](**distribution_config, name=name)
