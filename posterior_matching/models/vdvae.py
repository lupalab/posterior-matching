from collections import defaultdict
from typing import Tuple, Optional, Dict

import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from chex import Array
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax.math import reduce_logmeanexp


class PosteriorMatchingVDVAE(hk.Module):
    """A "Very Deep VAE", modified for Posterior Matching.

    This model is based on the VAE described in "Very Deep VAEs Generalize
    Autoregressive Models and Can Outperform Them on Images". This implementation is
    largely based on the original implementation found here:
    https://github.com/openai/vdvae
    This implementation is simply adapted to Jax and modified with a partial encoder
    and partial posteriors so that Posterior Matching can be performed.

    Args:
        image_shape: A tuple, (height, width, channels), specifying the input image
            shape.
        encoder_blocks: A string defining the structure of the encoder network.
        decoder_blocks: A string defining the structure of the decoder network.
        latent_dim: The dimensionality of the latent codes.
        width: The default number of channels in each convolutional layer.
        bottleneck_multiple: The relative size of the bottlenecks in the model.
        no_bias_above: The cutoff resolution above which an input bias will not be used.
        num_mixtures: The number of components in the discretized mixture of logistics
            distribution.
        custom_width_string: A string defining custom widths for each resolution.
        name: The optional name of the module.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        encoder_blocks: str,
        decoder_blocks: str,
        latent_dim: int = 16,
        width: int = 128,
        bottleneck_multiple: float = 0.25,
        no_bias_above: int = 64,
        num_mixtures: int = 10,
        custom_width_string: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        image_size = image_shape[0]
        num_channels = image_shape[-1]

        self.encoder = Encoder(
            width, encoder_blocks, bottleneck_multiple, custom_width_string
        )
        self.masked_encoder = Encoder(
            width, encoder_blocks, bottleneck_multiple, custom_width_string
        )
        self.decoder = PosteriorMatchingDecoder(
            latent_dim,
            image_size,
            num_channels,
            width,
            decoder_blocks,
            bottleneck_multiple,
            no_bias_above,
            num_mixtures,
            custom_width_string,
        )

    def __call__(self, x: Array, b: Array) -> Dict[str, Array]:
        activations = self.encoder(x / 127.5 - 1)
        masked_activations = self.masked_encoder(
            jnp.concatenate([(x / 127.5 - 1) * b, b], axis=-1)
        )

        px_z, stats = self.decoder.forward_posterior(activations, masked_activations)
        decoder_dist = self.decoder.out_net(px_z)

        pxz = decoder_dist.log_prob(x)
        kl = sum(d["kl"] for d in stats)
        pm_kl = sum(d["pm_kl"] for d in stats)

        return {
            "reconstruction_ll": pxz,
            "kl": kl,
            "pm_kl": pm_kl,
            "reconstruction": decoder_dist.mean(),
        }

    def is_log_probs(
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
        activations = self.encoder(x / 127.5 - 1)
        masked_activations = self.masked_encoder(
            jnp.concatenate([(x / 127.5 - 1) * b, b], axis=-1)
        )

        def sample_fn(c, _):
            px_z, pxo_z, stats = self.decoder.forward_lls(
                activations, masked_activations
            )
            px_z = self.decoder.out_net(px_z)
            pxo_z = self.decoder.out_net(pxo_z)

            pxz_ll = px_z.log_prob(x)
            pxoz_ll = einops.reduce(
                jnp.expand_dims(pxo_z.log_prob(x, independent=False), -1) * b,
                "b ... -> b",
                "sum",
            )

            pz = sum(d["pz"] for d in stats)
            qzx = sum(d["qzx"] for d in stats)
            masked_pz = sum(d["masked_pz"] for d in stats)
            masked_qzx = sum(d["masked_qzx"] for d in stats)

            px = pxz_ll + pz - qzx
            pxo = pxoz_ll + masked_pz - masked_qzx
            return c, (px, pxo)

        _, (px, pxo) = hk.scan(sample_fn, None, None, length=num_samples)

        px = reduce_logmeanexp(px, axis=0)
        pxo = reduce_logmeanexp(pxo, axis=0)
        pxu_xo = px - pxo

        return px, pxu_xo

    def sample(self, num_samples: int) -> Array:
        """Generates samples from the model.

        Args:
            num_samples: The number of samples to generate.

        Returns:
            The generated samples.
        """
        h = self.decoder.forward_prior(num_samples)
        decoder_dist = self.decoder.out_net(h)
        return decoder_dist.mean()

    def impute(self, x: Array, b: Array, num_samples: int = 100) -> Array:
        """Imputes unobserved features based on observed ones.

        Args:
            x: The observed feature values.
            b: The binary mask indicating which features are observed.
            num_samples: The number of imputations to generate per instance.

        Returns:
            A tensor of shape `[batch_size, num_samples, height, width, channels]`,
            which has imputed values at locations where `b == 0`.
        """

        def _impute_single(c, _):
            masked_activations = self.masked_encoder(
                jnp.concatenate([(x / 127.5 - 1) * b, b], axis=-1)
            )

            px_z, stats = self.decoder.forward_partial_posterior(masked_activations)
            decoder_dist = self.decoder.out_net(px_z)

            imputed = jnp.where(b == 1, x, decoder_dist.mean())
            return c, imputed

        _, imputations = hk.scan(_impute_single, None, None, length=num_samples)
        return einops.rearrange(imputations, "s b ... -> b s ...")


def get_3x3(out_dim):
    return hk.Conv2D(out_dim, 3, 1, padding=((1, 1), (1, 1)))


def get_1x1(out_dim, zero_last=False, init_multiple=None, in_dim=None):
    if zero_last:
        w_init = hk.initializers.Constant(0.0)
    elif init_multiple is not None:
        w_shape = (1, 1) + (in_dim, out_dim)
        fan_in_shape = np.prod(w_shape[:-1])
        stddev = 1.0 / np.sqrt(fan_in_shape)
        stddev *= init_multiple
        w_init = hk.initializers.TruncatedNormal(stddev=stddev)
    else:
        w_init = None

    return hk.Conv2D(out_dim, 1, 1, padding=((0, 0), (0, 0)), w_init=w_init)


def pad_channels(t, width):
    d = width - t.shape[-1]
    return jnp.pad(t, [(0, 0), (0, 0), (0, 0), (0, d)])


def parse_layer_string(s):
    layers = []
    for ss in s.split(","):
        if "x" in ss:
            res, num = ss.split("x")
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif "m" in ss:
            res, mixin = [int(a) for a in ss.split("m")]
            layers.append((res, mixin))
        elif "d" in ss:
            res, down_rate = [int(a) for a in ss.split("d")]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(",")
        for ss in s:
            k, v = ss.split(":")
            mapping[int(k)] = int(v)
    return mapping


class Block(hk.Module):
    def __init__(
        self,
        middle_width,
        out_width,
        down_rate=None,
        residual=False,
        use_3x3=True,
        zero_last=False,
        out_init_multiple=None,
        name=None,
    ):
        super().__init__(name=name)
        self.middle_width = middle_width
        self.out_width = out_width
        self.down_rate = down_rate
        self.residual = residual
        self.use_3x3 = use_3x3
        self.zero_last = zero_last
        self.out_init_multiple = out_init_multiple

    def __call__(self, x):
        c1 = get_1x1(self.middle_width)
        c2 = get_3x3(self.middle_width) if self.use_3x3 else get_1x1(self.middle_width)
        c3 = get_3x3(self.middle_width) if self.use_3x3 else get_1x1(self.middle_width)
        c4 = get_1x1(
            self.out_width,
            zero_last=self.zero_last,
            init_multiple=self.out_init_multiple,
            in_dim=self.middle_width,
        )

        h = c1(jax.nn.gelu(x))
        h = c2(jax.nn.gelu(h))
        h = c3(jax.nn.gelu(h))
        h = c4(jax.nn.gelu(h))

        out = x + h if self.residual else h

        if self.down_rate is not None:
            out = hk.AvgPool(self.down_rate, self.down_rate, padding="VALID")(out)

        return out


class Encoder(hk.Module):
    def __init__(
        self,
        width,
        blocks,
        bottleneck_multiple,
        custom_width_string=None,
        name=None,
    ):
        super().__init__(name=name)
        self.width = width
        self.widths = get_width_settings(width, custom_width_string)
        self.blocks = parse_layer_string(blocks)
        self.bottleneck_multiple = bottleneck_multiple

    def __call__(self, x):
        h = get_3x3(self.width)(x)

        activations = {}
        activations[h.shape[1]] = h

        for res, down_rate in self.blocks:
            use_3x3 = res > 2
            width = self.widths[res]
            h = Block(
                int(width * self.bottleneck_multiple),
                width,
                down_rate,
                residual=True,
                use_3x3=use_3x3,
                out_init_multiple=np.sqrt(1 / len(self.blocks)),
            )(h)

            res = h.shape[1]
            h = (
                h
                if h.shape[-1] == self.widths[res]
                else pad_channels(h, self.widths[res])
            )
            activations[res] = h

        return activations


class _LogisticMixtureDist(tfd.Distribution):
    def __init__(
        self,
        num_channels,
        component_logits,
        locs,
        scales,
        coeffs=None,
        low=0.0,
        high=255.0,
    ):
        super().__init__(jnp.float32, tfd.NOT_REPARAMETERIZED, False, False)
        self.num_channels = num_channels
        self.component_logits = component_logits
        self.locs = locs
        self.scales = scales
        self.coeffs = coeffs
        self.low = low
        self.high = high

    def log_prob(self, value, independent=True):
        if self.coeffs is not None:
            num_channels = self.num_channels
            num_coeffs = num_channels * (num_channels - 1) // 2
            transformed_value = (
                2.0 * (value - self.low) / (self.high - self.low)
            ) - 1.0

            loc_tensors = jnp.split(self.locs, num_channels, axis=-1)
            coef_tensors = jnp.split(self.coeffs, num_coeffs, axis=-1)
            channel_tensors = jnp.split(transformed_value, num_channels, axis=-1)

            coef_count = 0
            for i in range(num_channels):
                channel_tensors[i] = channel_tensors[i][..., jnp.newaxis, :]
                for j in range(i):
                    loc_tensors[i] += channel_tensors[j] * coef_tensors[coef_count]
                    coef_count += 1
            locs = jnp.concatenate(loc_tensors, axis=-1)
        else:
            locs = self.locs

        mixture_distribution = tfd.Categorical(logits=self.component_logits)

        locs = self.low + 0.5 * (self.high - self.low) * (locs + 1.0)
        scales = self.scales * 0.5 * (self.high - self.low)
        logistic_dist = tfd.QuantizedDistribution(
            distribution=tfd.TransformedDistribution(
                distribution=tfd.Logistic(loc=locs, scale=scales),
                bijector=tfb.Shift(shift=-0.5),
            ),
            low=self.low,
            high=self.high,
        )

        dist = tfd.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            components_distribution=tfd.Independent(
                logistic_dist, reinterpreted_batch_ndims=1
            ),
        )
        if independent:
            dist = tfd.Independent(dist, reinterpreted_batch_ndims=2)
        return dist.log_prob(value)

    def mean(self, name="mean", **kwargs):
        num_channels = self.num_channels
        weights = jnp.expand_dims(jax.nn.softmax(self.component_logits), axis=-1)
        loc_tensors = jnp.split(
            jnp.sum(self.locs * weights, axis=-2),
            num_channels,
            axis=-1,
        )
        scale_tensors = jnp.split(
            jnp.sum(self.scales * weights, axis=-2),
            num_channels,
            axis=-1,
        )

        if self.coeffs is not None:
            num_coeffs = num_channels * (num_channels - 1) // 2
            coef_tensors = jnp.split(
                jnp.sum(self.coeffs * weights, axis=-2),
                num_coeffs,
                axis=-1,
            )

        channel_samples = []
        coef_count = 0
        for i in range(num_channels):
            loc = loc_tensors[i]
            for c in channel_samples:
                loc += c * coef_tensors[coef_count]
                coef_count += 1

            logistic_samp = tfd.Logistic(loc=loc, scale=scale_tensors[i]).mean()
            logistic_samp = jnp.clip(logistic_samp, -1.0, 1.0)
            channel_samples.append(logistic_samp)

        out = jnp.concatenate(channel_samples, axis=-1)
        out = self.low + 0.5 * (self.high - self.low) * (out + 1.0)
        return jnp.round(out)


class LogisticMixture(hk.Module):
    """The discretized mixture of logistics distribution.

    See "PixelCNN++: Improving the PixelCNN with Discretized Logistic Mixture
    Likelihood and Other Modifications" for details.
    """

    def __init__(self, num_channels, num_mixtures, low=0, high=255, name=None):
        super().__init__(name=name)
        self.num_channels = num_channels
        self.num_mixtures = num_mixtures
        self.low = low
        self.high = high

    def __call__(self, x):
        num_coeffs = self.num_channels * (self.num_channels - 1) // 2
        num_out = self.num_channels * 2 + num_coeffs + 1
        params = hk.Conv2D(self.num_mixtures * num_out, 1, 1, padding="VALID")(x)
        params = jnp.reshape(params, [*x.shape[:-1], self.num_mixtures, num_out])

        if self.num_channels == 1:
            component_logits, locs, scales = jnp.split(params, 3, axis=-1)
            component_logits = jnp.squeeze(component_logits, axis=-1)
            scales = jax.nn.softplus(scales) + jnp.exp(-7.0)
            coeffs = None
        else:
            component_logits = params[..., :1]
            locs = params[..., 1 : self.num_channels + 1]
            scales = params[..., self.num_channels + 1 : 2 * self.num_channels + 1]
            coeffs = params[..., -num_coeffs:]
            component_logits = jnp.squeeze(component_logits, axis=-1)
            scales = jax.nn.softplus(scales) + jnp.exp(-7.0)

        return _LogisticMixtureDist(
            self.num_channels,
            component_logits,
            locs,
            scales,
            coeffs,
            self.low,
            self.high,
        )


class PosteriorMatchingDecoderBlock(hk.Module):
    def __init__(
        self,
        latent_dim,
        res,
        mixin,
        num_blocks,
        width,
        bottleneck_multiple,
        custom_width_string=None,
        name=None,
    ):
        super().__init__(name=name)
        self.base = res
        self.mixin = mixin
        self.widths = get_width_settings(width, custom_width_string)
        self.width = self.widths[res]
        self.latent_dim = latent_dim

        use_3x3 = res > 2

        self.posterior_block = Block(
            int(self.width * bottleneck_multiple),
            latent_dim * 2,
            residual=False,
            use_3x3=use_3x3,
        )
        self.masked_posterior_block = Block(
            int(self.width * bottleneck_multiple),
            latent_dim + latent_dim * (latent_dim + 1) // 2,
            residual=False,
            use_3x3=use_3x3,
        )
        self.prior_block = Block(
            int(self.width * bottleneck_multiple),
            latent_dim * 2 + self.width,
            residual=False,
            use_3x3=use_3x3,
            zero_last=True,
        )
        self.z_proj = get_1x1(
            self.width,
            init_multiple=np.sqrt(1 / num_blocks),
            in_dim=latent_dim,
        )
        self.resnet = Block(
            int(self.width * bottleneck_multiple),
            self.width,
            residual=True,
            use_3x3=use_3x3,
            out_init_multiple=np.sqrt(1 / num_blocks),
        )

    def sample_posterior(self, x, acts, masked_acts):
        post_loc, post_scale = jnp.split(
            self.posterior_block(jnp.concatenate([x, acts], axis=-1)), 2, axis=-1
        )
        masked_post_params = self.masked_posterior_block(
            jnp.concatenate([jax.lax.stop_gradient(x), masked_acts], axis=-1)
        )
        masked_post_loc = masked_post_params[..., : self.latent_dim]
        masked_post_scale = tfb.FillScaleTriL()(
            masked_post_params[..., self.latent_dim :]
        )
        posterior = tfd.Independent(
            tfd.MultivariateNormalDiag(post_loc, jax.nn.softplus(post_scale) + 1e-5)
        )
        posterior_no_grad = tfd.Independent(
            tfd.MultivariateNormalDiag(
                jax.lax.stop_gradient(post_loc),
                jax.lax.stop_gradient(jax.nn.softplus(post_scale) + 1e-5),
            )
        )
        masked_posterior = tfd.Independent(
            tfd.MultivariateNormalTriL(masked_post_loc, masked_post_scale)
        )

        prior_block_out = self.prior_block(x)
        h = prior_block_out[..., -self.width :]
        prior_loc, prior_scale = jnp.split(
            prior_block_out[..., : -self.width], 2, axis=-1
        )

        prior = tfd.Independent(
            tfd.MultivariateNormalDiag(prior_loc, jax.nn.softplus(prior_scale) + 1e-5)
        )

        x += h
        z = posterior.sample(seed=hk.next_rng_key())
        kl = posterior.kl_divergence(prior)
        pm_kl = posterior_no_grad.kl_divergence(masked_posterior)

        return z, x, kl, pm_kl

    def sample_partial_posterior(self, x, masked_acts):
        masked_post_params = self.masked_posterior_block(
            jnp.concatenate([x, masked_acts], axis=-1)
        )
        masked_post_loc = masked_post_params[..., : self.latent_dim]
        masked_post_scale = tfb.FillScaleTriL()(
            masked_post_params[..., self.latent_dim :]
        )
        masked_posterior = tfd.Independent(
            tfd.MultivariateNormalTriL(masked_post_loc, masked_post_scale)
        )

        prior_block_out = self.prior_block(x)
        h = prior_block_out[..., -self.width :]

        x += h
        z = masked_posterior.sample(seed=hk.next_rng_key())

        return z, x

    def sample_prior(self, x):
        prior_block_out = self.prior_block(x)
        h = prior_block_out[..., -self.width :]
        prior_loc, prior_scale = jnp.split(
            prior_block_out[..., : -self.width], 2, axis=-1
        )

        prior = tfd.Independent(
            tfd.MultivariateNormalDiag(prior_loc, jax.nn.softplus(prior_scale) + 1e-5)
        )

        x += h
        z = prior.sample(seed=hk.next_rng_key())

        return z, x

    def sample_lls(self, x, masked_x, acts, masked_acts):
        post_loc, post_scale = jnp.split(
            self.posterior_block(jnp.concatenate([x, acts], axis=-1)), 2, axis=-1
        )
        masked_post_params = self.masked_posterior_block(
            jnp.concatenate([masked_x, masked_acts], axis=-1)
        )
        masked_post_loc = masked_post_params[..., : self.latent_dim]
        masked_post_scale = tfb.FillScaleTriL()(
            masked_post_params[..., self.latent_dim :]
        )

        posterior = tfd.Independent(
            tfd.MultivariateNormalDiag(post_loc, jax.nn.softplus(post_scale) + 1e-5)
        )
        masked_posterior = tfd.Independent(
            tfd.MultivariateNormalTriL(masked_post_loc, masked_post_scale)
        )

        prior_block_out = self.prior_block(x)
        h = prior_block_out[..., -self.width :]
        prior_loc, prior_scale = jnp.split(
            prior_block_out[..., : -self.width], 2, axis=-1
        )
        prior = tfd.Independent(
            tfd.MultivariateNormalDiag(prior_loc, jax.nn.softplus(prior_scale) + 1e-5)
        )

        masked_prior_block_out = self.prior_block(masked_x)
        masked_h = masked_prior_block_out[..., -self.width :]
        masked_prior_loc, masked_prior_scale = jnp.split(
            masked_prior_block_out[..., : -self.width], 2, axis=-1
        )
        masked_prior = tfd.Independent(
            tfd.MultivariateNormalDiag(
                masked_prior_loc, jax.nn.softplus(masked_prior_scale) + 1e-5
            )
        )

        x += h
        masked_x += masked_h

        z = posterior.sample(seed=hk.next_rng_key())
        masked_z = masked_posterior.sample(seed=hk.next_rng_key())

        pz = prior.log_prob(z)
        qzx = posterior.log_prob(z)

        masked_pz = masked_prior.log_prob(masked_z)
        masked_qzx = masked_posterior.log_prob(masked_z)

        return z, masked_z, x, masked_x, pz, qzx, masked_pz, masked_qzx

    def get_inputs(self, xs, activations, masked_activations):
        acts = activations[self.base]
        masked_acts = masked_activations[self.base]
        try:
            x = xs[self.base]
        except KeyError:
            x = jnp.zeros_like(acts)
        if acts.shape[0] != x.shape[0]:
            x = jnp.repeat(x, acts.shape[0], axis=0)
        return x, acts, masked_acts

    def forward_posterior(self, xs, activations, masked_activations):
        x, acts, masked_acts = self.get_inputs(xs, activations, masked_activations)
        if self.mixin is not None:
            x += jax.image.resize(
                xs[self.mixin][..., : x.shape[-1]],
                x.shape,
                jax.image.ResizeMethod.NEAREST,
            )

        z, x, kl, pm_kl = self.sample_posterior(x, acts, masked_acts)
        x = x + self.z_proj(z)
        x = self.resnet(x)
        xs[self.base] = x

        return xs, dict(z=z, kl=kl, pm_kl=pm_kl)

    def forward_partial_posterior(self, xs, masked_activations):
        x, _, masked_acts = self.get_inputs(xs, masked_activations, masked_activations)
        if self.mixin is not None:
            x += jax.image.resize(
                xs[self.mixin][..., : x.shape[-1]],
                x.shape,
                jax.image.ResizeMethod.NEAREST,
            )

        z, x = self.sample_partial_posterior(x, masked_acts)
        x = x + self.z_proj(z)
        x = self.resnet(x)
        xs[self.base] = x

        return xs, dict(z=z)

    def forward_prior(self, xs):
        try:
            x = xs[self.base]
        except KeyError:
            ref = xs[list(xs.keys())[0]]
            x = jnp.zeros((ref.shape[0], self.base, self.base, self.widths[self.base]))
        if self.mixin is not None:
            x += jax.image.resize(
                xs[self.mixin][..., : x.shape[-1]],
                x.shape,
                jax.image.ResizeMethod.NEAREST,
            )

        z, x = self.sample_prior(x)
        x += self.z_proj(z)
        x = self.resnet(x)
        xs[self.base] = x

        return xs

    def forward_lls(self, xs, masked_xs, activations, masked_activations):
        x, acts, masked_acts = self.get_inputs(xs, activations, masked_activations)
        masked_x, _, _ = self.get_inputs(masked_xs, activations, masked_activations)
        if self.mixin is not None:
            x += jax.image.resize(
                xs[self.mixin][..., : x.shape[-1]],
                x.shape,
                jax.image.ResizeMethod.NEAREST,
            )
            masked_x += jax.image.resize(
                masked_xs[self.mixin][..., : masked_x.shape[-1]],
                masked_x.shape,
                jax.image.ResizeMethod.NEAREST,
            )

        z, masked_z, x, masked_x, pz, qzx, masked_pz, masked_qzx = self.sample_lls(
            x, masked_x, acts, masked_acts
        )
        x = x + self.z_proj(z)
        masked_x = masked_x + self.z_proj(masked_z)
        x = self.resnet(x)
        masked_x = self.resnet(masked_x)
        xs[self.base] = x
        masked_xs[self.base] = masked_x

        return (
            xs,
            masked_xs,
            dict(pz=pz, qzx=qzx, masked_pz=masked_pz, masked_qzx=masked_qzx),
        )


class PosteriorMatchingDecoder(hk.Module):
    def __init__(
        self,
        latent_dim,
        image_size,
        num_channels,
        width,
        blocks,
        bottleneck_multiple,
        no_bias_above,
        num_mixtures,
        custom_width_string=None,
        name=None,
    ):
        super().__init__(name=name)
        self.image_size = image_size
        self.widths = get_width_settings(width, custom_width_string)

        blocks = parse_layer_string(blocks)
        resolutions = set()
        self.blocks = []

        for res, mixin in blocks:
            self.blocks.append(
                PosteriorMatchingDecoderBlock(
                    latent_dim,
                    res,
                    mixin,
                    len(blocks),
                    width,
                    bottleneck_multiple,
                    custom_width_string,
                )
            )
            resolutions.add(res)

        self.resolutions = sorted(resolutions)

        self.bias_xs = [
            hk.get_parameter(
                f"x_bias_{res}]",
                (1, res, res, self.widths[res]),
                init=hk.initializers.Constant(0.0),
            )
            for res in self.resolutions
            if res <= no_bias_above
        ]

        self.out_net = LogisticMixture(num_channels, num_mixtures)

        self.gain = hk.get_parameter(
            "gain", (1, 1, 1, width), init=hk.initializers.Constant(1.0)
        )
        self.bias = hk.get_parameter(
            "bias", (1, 1, 1, width), init=hk.initializers.Constant(0.0)
        )
        self.final_fn = lambda x: x * self.gain + self.bias

    def forward_posterior(self, activations, masked_activations):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for block in self.blocks:
            xs, block_stats = block.forward_posterior(
                xs, activations, masked_activations
            )
            stats.append(block_stats)
        xs[self.image_size] = self.final_fn(xs[self.image_size])
        return xs[self.image_size], stats

    def forward_partial_posterior(self, masked_activations):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        for block in self.blocks:
            xs, block_stats = block.forward_partial_posterior(xs, masked_activations)
            stats.append(block_stats)
        xs[self.image_size] = self.final_fn(xs[self.image_size])
        return xs[self.image_size], stats

    def forward_prior(self, num_samples):
        xs = {}
        for bias in self.bias_xs:
            xs[bias.shape[2]] = jnp.repeat(bias, num_samples, axis=0)
        for block in self.blocks:
            xs = block.forward_prior(xs)
        xs[self.image_size] = self.final_fn(xs[self.image_size])
        return xs[self.image_size]

    def forward_lls(self, activations, masked_activations):
        stats = []
        xs = {a.shape[2]: a for a in self.bias_xs}
        masked_xs = {a.shape[2]: a for a in self.bias_xs}
        for block in self.blocks:
            xs, masked_xs, block_stats = block.forward_lls(
                xs, masked_xs, activations, masked_activations
            )
            stats.append(block_stats)
        xs[self.image_size] = self.final_fn(xs[self.image_size])
        masked_xs[self.image_size] = self.final_fn(masked_xs[self.image_size])
        return xs[self.image_size], masked_xs[self.image_size], stats
