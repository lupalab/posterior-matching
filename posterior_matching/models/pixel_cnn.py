"""This module implements the PixelCNN distribution.

This implementation is essentially just a rewrite of the TensorFlow probability
implementation, but in Jax.
"""
import functools
import math
import operator
from typing import Optional, Union, Sequence, Tuple

import distrax
import einops
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey, Array
from distrax._src.distributions.distribution import (
    IntLike,
    convert_seed_and_sample_shape,
)
from haiku._src.conv import compute_adjusted_padding
from tensorflow_probability.substrates.jax import distributions as tfd


class PixelCNN(distrax.Distribution):
    def __init__(
        self,
        num_indices,
        image_shape,
        dropout=0.5,
        num_resnet=15,
        num_hierarchies=1,
        num_filters=128,
        receptive_field_dims=(3, 3),
        name=None,
    ):
        self._event_shape = image_shape
        self._network = _PixelCNNNetwork(
            num_indices,
            dropout,
            num_resnet,
            num_hierarchies,
            num_filters,
            receptive_field_dims,
            name=name,
        )

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return self._event_shape

    def log_prob(
        self,
        value: Array,
        training=False,
        conditional_input=None,
    ) -> Array:
        dist = self._network(
            value, training=training, conditional_input=conditional_input
        )
        lls = dist.log_prob(value.astype(dist.dtype))
        return einops.reduce(lls, "b ... -> b", "sum")

    def logits(
        self,
        value: Array,
        training=False,
        conditional_input=None,
    ) -> Array:
        dist = self._network(
            value, training=training, conditional_input=conditional_input
        )
        return dist.logits

    def _sample_n(
        self,
        key: PRNGKey,
        n: int,
        conditional_input=None,
    ) -> Array:
        if conditional_input is None:

            def body_fun(i, state):
                key, x = state
                key, sample_key = jax.random.split(key, 2)
                dist = self._network(x, conditional_input=None)
                samples = dist.sample(seed=sample_key).astype(jnp.int32)
                m = jnp.reshape(
                    jax.nn.one_hot(i, math.prod(self._event_shape), dtype=jnp.bool_),
                    self._event_shape,
                )
                m = jnp.broadcast_to(m[jnp.newaxis, ...], x.shape)
                x = jnp.where(m, samples, x)
                return key, x

            init = (key, jnp.zeros((n, *self._event_shape), jnp.int32))
            _, samples = hk.fori_loop(0, math.prod(self._event_shape), body_fun, init)
            return samples
        else:

            def map_fun(key, cond):
                cond_input = jnp.tile(jnp.expand_dims(cond, 0), [n, 1])

                def body_fun(i, state):
                    key, x = state
                    key, sample_key = jax.random.split(key, 2)
                    dist = self._network(x, conditional_input=cond_input)
                    samples = dist.sample(seed=sample_key).astype(jnp.int32)
                    row, col = jnp.unravel_index(i, self._event_shape)
                    update = jax.lax.dynamic_slice(samples, [0, row, col], [n, 1, 1])
                    x = jax.lax.dynamic_update_slice(x, update, [0, row, col])
                    return key, x

                init = (key, jnp.zeros((n, *self._event_shape), jnp.int32))
                _, samples = hk.fori_loop(
                    0, math.prod(self._event_shape), body_fun, init
                )
                return samples

            samples = jax.vmap(map_fun)(
                jax.random.split(key, conditional_input.shape[0]), conditional_input
            )
            return einops.rearrange(samples, "b s ... -> s b ...")

    def sample(
        self,
        *,
        seed: Union[IntLike, PRNGKey],
        sample_shape: Union[IntLike, Sequence[IntLike]] = (),
        conditional_input=None,
    ) -> Array:
        rng, sample_shape = convert_seed_and_sample_shape(seed, sample_shape)
        num_samples = functools.reduce(operator.mul, sample_shape, 1)

        samples = self._sample_n(rng, num_samples, conditional_input=conditional_input)

        if sample_shape == ():
            samples = jnp.squeeze(samples, 0)

        return samples


# The `Conv` modules defined below are just small tweaks to the implementations in
# Haiku that allow the kernel masks used by PixelCNN to be more easily specified.


class _ConvND(hk.ConvND):
    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        precision: Optional[jax.lax.Precision] = None,
    ) -> jnp.ndarray:
        unbatched_rank = self.num_spatial_dims + 1
        allowed_ranks = [unbatched_rank, unbatched_rank + 1]
        if inputs.ndim not in allowed_ranks:
            raise ValueError(
                f"Input to ConvND needs to have rank in {allowed_ranks},"
                f" but input has shape {inputs.shape}."
            )

        unbatched = inputs.ndim == unbatched_rank
        if unbatched:
            inputs = jnp.expand_dims(inputs, axis=0)

        if inputs.shape[self.channel_index] % self.feature_group_count != 0:
            raise ValueError(
                f"Inputs channels {inputs.shape[self.channel_index]} "
                f"should be a multiple of feature_group_count "
                f"{self.feature_group_count}"
            )
        w_shape = self.kernel_shape + (
            inputs.shape[self.channel_index] // self.feature_group_count,
            self.output_channels,
        )

        w_init = self.w_init
        if w_init is None:
            fan_in_shape = np.prod(w_shape[:-1])
            stddev = 1.0 / np.sqrt(fan_in_shape)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)

        if self.mask is not None:
            w *= self.mask

        out = jax.lax.conv_general_dilated(
            inputs,
            w,
            window_strides=self.stride,
            padding=self.padding,
            lhs_dilation=self.lhs_dilation,
            rhs_dilation=self.kernel_dilation,
            dimension_numbers=self.dimension_numbers,
            feature_group_count=self.feature_group_count,
            precision=precision,
        )

        if self.with_bias:
            if self.channel_index == -1:
                bias_shape = (self.output_channels,)
            else:
                bias_shape = (self.output_channels,) + (1,) * self.num_spatial_dims
            b = hk.get_parameter("b", bias_shape, inputs.dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        if unbatched:
            out = jnp.squeeze(out, axis=0)
        return out


class _Conv2D(_ConvND):
    def __init__(
        self,
        output_channels: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[
            str, Sequence[Tuple[int, int]], hk.pad.PadFn, Sequence[hk.pad.PadFn]
        ] = "SAME",
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        data_format: str = "NHWC",
        mask: Optional[jnp.ndarray] = None,
        feature_group_count: int = 1,
        name: Optional[str] = None,
    ):
        super().__init__(
            num_spatial_dims=2,
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=stride,
            rate=rate,
            padding=padding,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            data_format=data_format,
            mask=mask,
            feature_group_count=feature_group_count,
            name=name,
        )


class _ConvNDTranspose(hk.ConvNDTranspose):
    def __call__(
        self,
        inputs: jnp.ndarray,
        *,
        precision: Optional[jax.lax.Precision] = None,
    ) -> jnp.ndarray:

        unbatched_rank = self.num_spatial_dims + 1
        allowed_ranks = [unbatched_rank, unbatched_rank + 1]
        if inputs.ndim not in allowed_ranks:
            raise ValueError(
                f"Input to ConvNDTranspose needs to have rank in "
                f"{allowed_ranks}, but input has shape {inputs.shape}."
            )

        unbatched = inputs.ndim == unbatched_rank
        if unbatched:
            inputs = jnp.expand_dims(inputs, axis=0)

        input_channels = inputs.shape[self.channel_index]
        w_shape = self.kernel_shape + (self.output_channels, input_channels)

        w_init = self.w_init
        if w_init is None:
            fan_in_shape = self.kernel_shape + (input_channels,)
            stddev = 1.0 / np.sqrt(np.prod(fan_in_shape))
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", w_shape, inputs.dtype, init=w_init)

        if self.mask is not None:
            w = w * self.mask

        padding = self.padding
        if self.output_shape is not None:
            input_shape = (
                inputs.shape[2:] if self.channel_index == 1 else inputs.shape[1:-1]
            )
            padding = tuple(
                map(
                    lambda i, o, k, s: compute_adjusted_padding(
                        i, o, k, s, self.padding
                    ),
                    input_shape,
                    self.output_shape,
                    self.kernel_shape,
                    self.stride,
                )
            )

        out = jax.lax.conv_transpose(
            inputs,
            w,
            strides=self.stride,
            padding=padding,
            dimension_numbers=self.dimension_numbers,
            precision=precision,
        )

        if self.with_bias:
            if self.channel_index == -1:
                bias_shape = (self.output_channels,)
            else:
                bias_shape = (self.output_channels,) + (1,) * self.num_spatial_dims
            b = hk.get_parameter("b", bias_shape, inputs.dtype, init=self.b_init)
            b = jnp.broadcast_to(b, out.shape)
            out = out + b

        if unbatched:
            out = jnp.squeeze(out, axis=0)
        return out


class _Conv2DTranspose(_ConvNDTranspose):
    def __init__(
        self,
        output_channels: int,
        kernel_shape: Union[int, Sequence[int]],
        stride: Union[int, Sequence[int]] = 1,
        output_shape: Optional[Union[int, Sequence[int]]] = None,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        data_format: str = "NHWC",
        mask: Optional[jnp.ndarray] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            num_spatial_dims=2,
            output_channels=output_channels,
            kernel_shape=kernel_shape,
            stride=stride,
            output_shape=output_shape,
            padding=padding,
            with_bias=with_bias,
            w_init=w_init,
            b_init=b_init,
            data_format=data_format,
            mask=mask,
            name=name,
        )


class _PixelCNNNetwork(hk.Module):
    def __init__(
        self,
        num_indices,
        dropout=0.5,
        num_resnet=5,
        num_hierarchies=3,
        num_filters=160,
        receptive_field_dims=(3, 3),
        name=None,
    ):
        super().__init__(name=name)
        self._num_indices = num_indices
        self._dropout = dropout
        self._num_resnet = num_resnet
        self._num_hierarchies = num_hierarchies
        self._num_filters = num_filters
        self._receptive_field_dims = receptive_field_dims

    def __call__(self, image_input, conditional_input=None, training=False):
        def concat_elu(x):
            return jax.nn.elu(jnp.concatenate([x, -x], axis=-1))

        Conv2D = functools.partial(
            _Conv2D, output_channels=self._num_filters, padding="SAME"
        )

        Conv2DTranspose = functools.partial(
            _Conv2DTranspose,
            output_channels=self._num_filters,
            padding="SAME",
            stride=(2, 2),
        )

        dropout_rate = self._dropout * training

        rows, cols = self._receptive_field_dims

        kernel_valid_dims = {
            "vertical": (rows - 1, cols),
            "horizontal": (2, cols // 2 + 1),
        }

        kernel_sizes = {"vertical": (2 * rows - 3, cols), "horizontal": (3, cols)}

        kernel_constraints = {
            k: _make_kernel_constraint(kernel_sizes[k], (0, v[0]), (0, v[1]))
            for k, v in kernel_valid_dims.items()
        }

        image_input = hk.Embed(self._num_indices, self._num_filters)(image_input)

        vertical_stack_init = Conv2D(
            kernel_shape=(2 * rows - 1, cols),
            mask=_make_kernel_constraint(
                (2 * rows - 1, cols), (0, rows - 1), (0, cols)
            ),
        )(image_input)

        horizontal_stack_up = Conv2D(
            kernel_shape=(3, cols),
            mask=_make_kernel_constraint((3, cols), (0, 1), (0, cols)),
        )(image_input)

        horizontal_stack_left = Conv2D(
            kernel_shape=(3, cols),
            mask=_make_kernel_constraint((3, cols), (0, 2), (0, cols // 2)),
        )(image_input)

        horizontal_stack_init = horizontal_stack_up + horizontal_stack_left

        layer_stacks = {
            "vertical": [vertical_stack_init],
            "horizontal": [horizontal_stack_init],
        }

        for i in range(self._num_hierarchies):
            for _ in range(self._num_resnet):
                for stack in ["vertical", "horizontal"]:
                    input_x = layer_stacks[stack][-1]
                    x = concat_elu(input_x)
                    x = Conv2D(
                        kernel_shape=kernel_sizes[stack], mask=kernel_constraints[stack]
                    )(x)

                    if stack == "horizontal":
                        h = concat_elu(layer_stacks["vertical"][-1])
                        h = hk.Linear(self._num_filters)(h)
                        x += h

                    x = concat_elu(x)
                    x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
                    x = Conv2D(
                        output_channels=2 * self._num_filters,
                        kernel_shape=kernel_sizes[stack],
                        mask=kernel_constraints[stack],
                    )(x)

                    if conditional_input is not None:
                        h_projection = _build_and_apply_h_projection(
                            conditional_input, self._num_filters
                        )
                        x += h_projection

                    x = _apply_sigmoid_gating(x)

                    out = input_x + x
                    layer_stacks[stack].append(out)

            if i < self._num_hierarchies - 1:
                for stack in ["vertical", "horizontal"]:
                    x = layer_stacks[stack][-1]
                    h, w = kernel_valid_dims[stack]
                    kernel_height = 2 * h
                    if stack == "vertical":
                        kernel_width = w + 1
                    else:
                        kernel_width = 2 * w

                    kernel_size = (kernel_height, kernel_width)
                    kernel_constraint = _make_kernel_constraint(
                        kernel_size, (0, h), (0, w)
                    )
                    x = Conv2D(
                        stride=(2, 2),
                        kernel_shape=kernel_size,
                        mask=kernel_constraint,
                    )(x)
                    layer_stacks[stack].append(x)

        upward_pass = {key: stack.pop() for key, stack in layer_stacks.items()}

        for i in range(self._num_hierarchies):
            num_resnet = self._num_resnet if i == 0 else self._num_resnet + 1

            for _ in range(num_resnet):
                for stack in ["vertical", "horizontal"]:
                    input_x = upward_pass[stack]
                    x_symmetric = layer_stacks[stack].pop()

                    x = concat_elu(input_x)
                    x = Conv2D(
                        kernel_shape=kernel_sizes[stack], mask=kernel_constraints[stack]
                    )(x)

                    if stack == "horizontal":
                        x_symmetric = jnp.concatenate(
                            [upward_pass["vertical"], x_symmetric], axis=-1
                        )

                    h = concat_elu(x_symmetric)
                    h = hk.Linear(self._num_filters)(h)
                    x += h

                    x = concat_elu(x)
                    x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
                    x = Conv2D(
                        output_channels=2 * self._num_filters,
                        kernel_shape=kernel_sizes[stack],
                        mask=kernel_constraints[stack],
                    )(x)

                    if conditional_input is not None:
                        h_projection = _build_and_apply_h_projection(
                            conditional_input, self._num_filters
                        )
                        x += h_projection

                    x = _apply_sigmoid_gating(x)
                    upward_pass[stack] = input_x + x

            if i < self._num_hierarchies - 1:
                for stack in ["vertical", "horizontal"]:
                    h, w = kernel_valid_dims[stack]
                    kernel_height = 2 * h - 2
                    if stack == "vertical":
                        kernel_width = w + 1
                        kernel_constraint = _make_kernel_constraint(
                            (kernel_height, kernel_width),
                            (h - 2, kernel_height),
                            (0, w),
                        )
                    else:
                        kernel_width = 2 * w - 2
                        kernel_constraint = _make_kernel_constraint(
                            (kernel_height, kernel_width),
                            (h - 2, kernel_height),
                            (w - 2, kernel_width),
                        )

                    x = upward_pass[stack]
                    x = Conv2DTranspose(
                        kernel_shape=(kernel_height, kernel_width),
                        mask=kernel_constraint,
                    )(x)
                    upward_pass[stack] = x

        x_out = jax.nn.elu(upward_pass["horizontal"])

        params = Conv2D(output_channels=self._num_indices, kernel_shape=1)(x_out)
        return tfd.Categorical(params)


def _make_kernel_constraint(kernel_size, valid_rows, valid_columns):
    mask = np.zeros(kernel_size)
    lower, upper = valid_rows
    left, right = valid_columns
    mask[lower:upper, left:right] = 1.0
    mask = mask[:, :, np.newaxis, np.newaxis]
    return mask


def _build_and_apply_h_projection(h, num_filters):
    h = einops.rearrange(h, "b ... -> b (...)")
    h_projection = hk.Linear(2 * num_filters, w_init=hk.initializers.RandomNormal())(h)
    return h_projection[..., jnp.newaxis, jnp.newaxis, :]


def _apply_sigmoid_gating(x):
    activation_tensor, gate_tensor = jnp.split(x, 2, axis=-1)
    sigmoid_gate = jax.nn.sigmoid(gate_tensor)
    return sigmoid_gate * activation_tensor
