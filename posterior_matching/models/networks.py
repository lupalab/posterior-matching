from typing import Sequence, Tuple, Optional, Callable, Any, Dict

import chex
import einops
import haiku as hk
import jax


class ConvEncoder(hk.Module):
    """A simple convolutional encoder network.

    Args:
        conv_layers: A list of tuples, where each tuples has the form
            (num_filters, kernel_size, stride).
        name: The optional name of the module.
    """

    def __init__(
        self, conv_layers: Sequence[Tuple[int, int, int]], name: Optional[str] = None
    ):
        super().__init__(name=name)
        self._conv_layers = conv_layers

    def __call__(self, x, is_training=False):
        chex.assert_rank(x, 4)

        h = x

        for i, (filters, kernel, strides) in enumerate(self._conv_layers):
            h = hk.Conv2D(
                output_channels=filters,
                kernel_shape=kernel,
                stride=strides,
                padding="VALID" if i == len(self._conv_layers) - 1 else "SAME",
            )(h)
            h = jax.nn.leaky_relu(h)

        return h


class ConvDecoder(hk.Module):
    """A simple convolutional decoder network.

    Args:
        conv_layers: A list of tuples, where each tuples has the form
            (num_filters, kernel_size, stride).
        name: The optional name of the module.
    """

    def __init__(
        self, conv_layers: Sequence[Tuple[int, int, int]], name: Optional[str] = None
    ):
        super().__init__(name=name)
        self._conv_layers = conv_layers

    def __call__(self, x, is_training=False):
        chex.assert_rank(x, 2)

        h = einops.rearrange(x, "b z -> b 1 1 z")

        for i, (filters, kernel, strides) in enumerate(self._conv_layers):
            h = hk.Conv2DTranspose(
                output_channels=filters,
                kernel_shape=kernel,
                stride=strides,
                padding="VALID" if i == 0 else "SAME",
            )(h)
            h = jax.nn.leaky_relu(h)

        chex.assert_rank(h, 4)

        return h


class ResidualMLP(hk.Module):
    """A multi-layer perception with residual connections.

    Args:
        residual_blocks: The number of residual blocks.
        hidden_units: The number of hidden units in each layer.
        activation: The activation function to apply after each layer.
        activate_final: Whether or not to apply the activation function to the network's
            output.
        dropout: The dropout rate.
        w_init: The weight initializer to use.
        layer_norm: Whether or not to use layer norm.
        name: The optional name of the module.
    """

    def __init__(
        self,
        residual_blocks: int = 2,
        hidden_units: int = 256,
        activation: Callable = jax.nn.relu,
        activate_final: bool = True,
        dropout: float = 0.0,
        w_init: hk.initializers.Initializer = None,
        layer_norm: bool = False,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)

        self._residual_blocks = residual_blocks
        self._hidden_units = hidden_units
        self._activation = activation
        self._activate_final = activate_final
        self._dropout = dropout
        self._w_init = w_init
        self._layer_norm = layer_norm

    def __call__(self, x, is_training=False):
        chex.assert_rank(x, 2)

        dropout_rate = self._dropout if is_training else 0.0

        h = hk.Linear(self._hidden_units, w_init=self._w_init)(x)
        if self._layer_norm:
            h = hk.LayerNorm(-1, False, False)(h)

        for _ in range(self._residual_blocks):
            res = self._activation(h)
            res = hk.Linear(self._hidden_units, w_init=self._w_init)(res)
            if self._layer_norm:
                res = hk.LayerNorm(-1, False, False)(res)
            res = self._activation(res)
            res = hk.dropout(hk.next_rng_key(), dropout_rate, res)
            res = hk.Linear(self._hidden_units, w_init=self._w_init)(res)
            if self._layer_norm:
                res = hk.LayerNorm(-1, False, False)(res)
            h += res

        if self._activate_final:
            h = self._activation(h)

        return h


_NETWORKS = {
    "ConvEncoder": ConvEncoder,
    "ConvDecoder": ConvDecoder,
    "ResidualMLP": ResidualMLP,
}


def get_network(
    network_type: str,
    network_config: Optional[Dict[str, Any]] = None,
    name: Optional[str] = None,
) -> hk.Module:
    """Creates a network based on the specified type and config.

    Args:
        network_type: The type of network to create.
        network_config: A dictionary of keyword arguments to pass to the network
            upon construction.
        name: The optional name of the module.

    Returns:
        The specified network, as an `hk.Module`.
    """
    network_config = network_config or {}
    return _NETWORKS[network_type](**network_config, name=name)
