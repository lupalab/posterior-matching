from typing import Optional, Dict, Any

import einops
import haiku as hk
import jax
import jax.numpy as jnp
from chex import Array
from tensorflow_probability.substrates.jax import distributions as tfd

from posterior_matching.models.pixel_cnn import PixelCNN


class VQVAE(hk.Module):
    """The Vector-Quantized VAE model.

    This is the model described in "Neural Discrete Representation Learning".

    Args:
        output_channels: The number of channels the model's output should have.
        embedding_dim: The dimensionality of the vectors in the codebook.
        num_embeddings: The number of vectors in the codebook.
        hidden_units: The baseline number of filters in each convolutional layer.
        residual_blocks: The number of residual blocks.
        residual_hidden_units: The number of filters in the first layer of each
            residual block.
        decay: The decay rate when using EMA to learn the quantized codes.
        commitment_cost: The coefficient for the commitment term in the VQ loss.
        cross_replica_axis: The pmap axis to compute the EMA over.
        use_ema: Whether or not to use the EMA version of VQ-VAE.
        name: The optional name of the module.
    """

    def __init__(
        self,
        output_channels: int = 3,
        embedding_dim: int = 64,
        num_embeddings: int = 512,
        hidden_units: int = 128,
        residual_blocks: int = 2,
        residual_hidden_units: int = 128,
        decay: float = 0.99,
        commitment_cost: float = 0.25,
        cross_replica_axis: Optional[str] = None,
        use_ema: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self.encoder = hk.Sequential(
            [
                ConvResidualEncoder(
                    hidden_units, residual_blocks, residual_hidden_units
                ),
                hk.Conv2D(embedding_dim, 1, 1, name="pre_vq_conv"),
            ]
        )

        self.decoder = ConvResidualDecoder(
            hidden_units,
            residual_blocks,
            residual_hidden_units,
            output_channels,
        )

        if use_ema:
            self.vq = hk.nets.VectorQuantizerEMA(
                embedding_dim,
                num_embeddings,
                commitment_cost,
                decay,
                cross_replica_axis=cross_replica_axis,
            )
        else:
            self.vq = hk.nets.VectorQuantizer(
                embedding_dim, num_embeddings, commitment_cost
            )

    def __call__(self, inputs: Array, is_training: bool = False) -> Dict[str, Array]:
        z = self.encoder(inputs)
        vq_output = self.vq(z, is_training=is_training)
        decoder_dist = self.decoder(vq_output["quantize"])

        reconstruction_loss = -jnp.mean(
            einops.reduce(decoder_dist.log_prob(inputs), "b ... -> b", "sum")
        )

        loss = reconstruction_loss + vq_output["loss"]

        return {
            "loss": loss,
            "vq_output": vq_output,
            "z": z,
            "reconstruction": decoder_dist.mean(),
            "reconstruction_loss": reconstruction_loss,
            "decoder_dist": decoder_dist,
        }


class VQVAEPartialEncoder(hk.Module):
    """A partial encoder for VQ-VAE with Posterior Matching.

    Args:
        conditional_dim: The dimensionality of the outputs of this model, i.e. of the
            conditioning vectors.
        vqvae_config: The config for the VQ-VAE model that this partial encoder is being
            used with.
        name: The optional name of the module.
    """

    def __init__(
        self,
        conditional_dim: int,
        vqvae_config: Dict[str, Any],
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._net = hk.Sequential(
            [
                ConvResidualEncoder(
                    vqvae_config["hidden_units"],
                    vqvae_config["residual_blocks"],
                    vqvae_config["residual_hidden_units"],
                ),
                hk.Flatten(),
                hk.Linear(conditional_dim),
            ]
        )

    def __call__(self, *args, **kwargs):
        return self._net(*args, **kwargs)


class ConvResidualStack(hk.Module):
    def __init__(
        self,
        hidden_units,
        residual_blocks,
        residual_hidden_units,
        activate_final=True,
        name=None,
    ):
        super(ConvResidualStack, self).__init__(name=name)
        self._hidden_units = hidden_units
        self._residual_blocks = residual_blocks
        self._residual_hidden_units = residual_hidden_units
        self._activate_final = activate_final

    def __call__(self, inputs):
        layers = []
        for i in range(self._residual_blocks):
            conv3 = hk.Conv2D(
                output_channels=self._residual_hidden_units,
                kernel_shape=(3, 3),
                stride=(1, 1),
                name="res3x3_%d" % i,
            )
            conv1 = hk.Conv2D(
                output_channels=self._hidden_units,
                kernel_shape=(1, 1),
                stride=(1, 1),
                name="res1x1_%d" % i,
            )
            layers.append((conv3, conv1))

        h = inputs
        for conv3, conv1 in layers:
            conv3_out = conv3(jax.nn.relu(h))
            conv1_out = conv1(jax.nn.relu(conv3_out))
            h += conv1_out

        if self._activate_final:
            h = jax.nn.relu(h)

        return h


class ConvResidualEncoder(hk.Module):
    def __init__(self, hidden_units, residual_blocks, residual_hidden_units, name=None):
        super(ConvResidualEncoder, self).__init__(name=name)
        self._hidden_units = hidden_units
        self._residual_blocks = residual_blocks
        self._residual_hidden_units = residual_hidden_units

    def __call__(self, x):
        enc_1 = hk.Conv2D(
            output_channels=self._hidden_units // 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_1",
        )
        enc_2 = hk.Conv2D(
            output_channels=self._hidden_units,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="enc_2",
        )
        enc_3 = hk.Conv2D(
            output_channels=self._hidden_units,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="enc_3",
        )
        residual_stack = ConvResidualStack(
            self._hidden_units, self._residual_blocks, self._residual_hidden_units
        )

        h = jax.nn.relu(enc_1(x))
        h = jax.nn.relu(enc_2(h))
        h = jax.nn.relu(enc_3(h))
        return residual_stack(h)


class ConvResidualDecoder(hk.Module):
    def __init__(
        self,
        hidden_units,
        residual_blocks,
        residual_hidden_units,
        output_channels,
        name=None,
    ):
        super(ConvResidualDecoder, self).__init__(name=name)
        self._hidden_units = hidden_units
        self._residual_blocks = residual_blocks
        self._residual_hidden_units = residual_hidden_units
        self._output_channels = output_channels

    def __call__(self, z, scale=None):
        residual_stack = ConvResidualStack(
            self._hidden_units, self._residual_blocks, self._residual_hidden_units
        )
        dec_1 = hk.Conv2D(
            output_channels=self._hidden_units,
            kernel_shape=(3, 3),
            stride=(1, 1),
            name="dec_1",
        )
        dec_2 = hk.Conv2DTranspose(
            output_channels=self._hidden_units // 2,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_2",
        )
        dec_3 = hk.Conv2DTranspose(
            output_channels=self._output_channels,
            kernel_shape=(4, 4),
            stride=(2, 2),
            name="dec_3",
        )

        h = dec_1(z)
        h = residual_stack(h)
        h = jax.nn.relu(dec_2(h))
        params = dec_3(h)

        if scale is None:
            scale = (
                jnp.exp(
                    hk.get_parameter(
                        "log_scale", [], init=hk.initializers.Constant(0.0)
                    )
                )
                + 1e-5
            )

        return tfd.Normal(params, scale)


def vqvae_impute(
    vqvae: VQVAE,
    partial_encoder: VQVAEPartialEncoder,
    partial_posterior: PixelCNN,
    x: Array,
    b: Array,
    num_samples: int = 5,
) -> Array:
    """Performs imputation for a VQ-VAE model and it's partially observed posterior.

    Args:
        vqvae: The base VQ-VAE model.
        partial_encoder: The partial encoder network.
        partial_posterior: The conditional PixelCNN++ that represents the partially
            observed posterior.
        x: The observed feature values.
        b: The binary mask indicating which features are observed.
        num_samples: The number of imputations to generate per instance.

    Returns:
        A tensor of shape `[batch_size, num_samples, height, width, channels]`,
        which has imputed values at locations where `b == 0`.
    """
    x_o_b = jnp.concatenate([x * b, b], axis=-1)
    cond_latents = partial_encoder(x_o_b)

    samples = partial_posterior.sample(
        sample_shape=num_samples, seed=hk.next_rng_key(), conditional_input=cond_latents
    )

    def decoder_mean(q):
        return vqvae.decoder(q).mean()

    quantized = jax.vmap(vqvae.vq.quantize)(samples)
    imputations = jax.vmap(decoder_mean)(quantized)
    imputations = einops.rearrange(imputations, "s b ... -> b s ...")

    imputations = jnp.where(
        b[:, jnp.newaxis, ...],
        x[:, jnp.newaxis, ...],
        imputations,
    )

    return jnp.clip(imputations, 0.0, 1.0)
