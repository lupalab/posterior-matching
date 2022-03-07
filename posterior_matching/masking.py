from abc import ABC, abstractmethod
from typing import Optional, Tuple, Sequence, Union

import numpy as np
import tensorflow as tf
from PIL import Image


class MaskGenerator(ABC):
    def __init__(
        self, seed: Optional[int] = None, dtype: Union[str, object] = np.float32
    ):
        self._rng = np.random.RandomState(seed=seed)
        self._dtype = dtype

    def __call__(self, shape: Sequence[int]):
        return self.call(np.asarray(shape)).astype(self._dtype)

    @abstractmethod
    def call(self, shape: Sequence[int]) -> np.ndarray:
        pass


class MixtureMaskGenerator(MaskGenerator):
    def __init__(self, generators, weights=None, batch_level=False, **kwargs):
        super().__init__(**kwargs)
        self._generators = generators

        if weights is None:
            weights = [1] * len(generators)
        weights = np.asarray(weights)

        assert len(generators) == len(weights)

        self._weights = weights / np.sum(weights)

        self._batch_level = batch_level

    def call(self, shape, **kwargs):
        if self._batch_level:
            ind = self._rng.choice(len(self._generators), 1, p=self._weights)[0]
            return self._generators[ind](shape)

        inds = self._rng.choice(len(self._generators), shape[0], p=self._weights)
        return np.concatenate(
            [self._generators[i]((1, *shape[1:])) for i in inds], axis=0
        )


class UniformMaskGenerator(MaskGenerator):
    def __init__(self, bounds: Optional[Tuple[float, float]] = None, **kwargs):
        super().__init__(**kwargs)
        self._bounds = bounds

    def call(self, shape: Sequence[int]) -> np.ndarray:
        orig_shape = None
        if len(shape) != 2:
            orig_shape = shape
            shape = (shape[0], np.prod(shape[1:]))

        b, d = shape

        result = []
        for _ in range(b):
            if self._bounds is None:
                q = self._rng.choice(d)
            else:
                l = int(d * self._bounds[0])
                h = int(d * self._bounds[1])
                q = l + self._rng.choice(h)
            inds = self._rng.choice(d, q, replace=False)
            mask = np.zeros(d)
            mask[inds] = 1
            result.append(mask)

        result = np.vstack(result)

        if orig_shape is not None:
            result = np.reshape(result, orig_shape)

        return result


class BernoulliMaskGenerator(MaskGenerator):
    def __init__(self, p: float = 0.5, **kwargs):
        super().__init__(**kwargs)

        self.p = p

    def call(self, shape: Sequence[int]) -> np.ndarray:
        return self._rng.binomial(1, self.p, size=shape)


class ImageBernoulliMaskGenerator(MaskGenerator):
    def __init__(self, p=0.2, **kwargs):
        super().__init__(**kwargs)

        self.p = p

    def call(self, shape, **kwargs):
        assert (
            len(shape) == 4
        ), f"expected shape of size [batch_dim, height, width, channels], got {shape}"
        return self._rng.binomial(1, self.p, size=[*shape[:-1], 1])


class RectangleMaskGenerator(MaskGenerator):
    def __init__(self, min_prop=0.3, max_prop=1.0, **kwargs):
        super().__init__(**kwargs)
        self._min_prop = min_prop
        self._max_prop = max_prop

    def _random_rectangle_coordinates(self, width, height):
        x1, x2 = self._rng.randint(0, width, 2)
        y1, y2 = self._rng.randint(0, height, 2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return int(x1), int(y1), int(x2), int(y2)

    def call(self, shape, **kwargs):
        assert (
            len(shape) == 4
        ), f"expected shape of size [batch_dim, height, width, channels], got {shape}"
        batch_size, height, width, num_channels = shape
        batch_result = []

        for _ in range(batch_size):
            mask = np.ones((height, width, 1))
            x1, y1, x2, y2 = self._random_rectangle_coordinates(width, height)
            sqr = width * height
            while not (
                self._min_prop * sqr
                <= (x2 - x1 + 1) * (y2 - y1 + 1)
                <= self._max_prop * sqr
            ):
                x1, y1, x2, y2 = self._random_rectangle_coordinates(width, height)
            mask[y1 : y2 + 1, x1 : x2 + 1, :] = 0
            batch_result.append(mask)

        return np.array(batch_result)


class FixedRectangleMaskGenerator(MaskGenerator):
    def __init__(self, y1, x1, y2, x2, **kwargs):
        super().__init__(**kwargs)
        self.y1 = y1
        self.x1 = x1
        self.y2 = y2
        self.x2 = x2

    def call(self, shape, **kwargs):
        assert (
            len(shape) == 4
        ), f"expected shape of size [batch_dim, height, width, channels], got {shape}"
        mask = np.ones((*shape[:-1], 1))
        mask[:, self.y1 : self.y2, self.x1 : self.x2, :] = 0
        return mask


class SquareMaskGenerator(MaskGenerator):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def call(self, shape, **kwargs):
        assert (
            len(shape) == 4
        ), f"expected shape of size [batch_dim, height, width, channels], got {shape}"
        _, height, width, _ = shape
        mask = np.ones((*shape[:-1], 1))
        x = self._rng.randint(width - self.size)
        y = self._rng.randint(height - self.size)
        mask[:, y : y + self.size, x : x + self.size, :] = 0
        return mask


class RandomPatternMaskGenerator(MaskGenerator):
    def __init__(
        self, max_size=10000, resolution=0.06, density=0.25, update_freq=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.max_size = max_size
        self.resolution = resolution
        self.density = density
        self.update_freq = update_freq
        self.pattern = None
        self.points_used = None

        self._regenerate_cache()

    def _regenerate_cache(self):
        low_size = int(self.resolution * self.max_size)
        low_pattern = self._rng.uniform(0, 1, size=(low_size, low_size))
        low_pattern = low_pattern.astype("float32")
        pattern = Image.fromarray(low_pattern)
        pattern = pattern.resize((self.max_size, self.max_size), Image.BICUBIC)
        pattern = np.array(pattern)
        pattern = (pattern < self.density).astype("float32")
        self.pattern = pattern
        self.points_used = 0

    def call(self, shape, density_std=0.05):
        assert (
            len(shape) == 4
        ), "expected shape of size [batch_dim, height, width, channels]"
        batch_size, height, width, _ = shape
        batch_result = []

        for _ in range(batch_size):
            x = self._rng.randint(0, self.max_size - width + 1)
            y = self._rng.randint(0, self.max_size - height + 1)
            res = self.pattern[y : y + height, x : x + width]
            coverage = res.mean()

            while not (
                self.density - density_std < coverage < self.density + density_std
            ):
                x = self._rng.randint(0, self.max_size - width + 1)
                y = self._rng.randint(0, self.max_size - height + 1)
                res = self.pattern[y : y + height, x : x + width]
                coverage = res.mean()

            mask = res[:, :, None]
            mask = 1.0 - mask
            self.points_used += width * height

            if self.update_freq * (self.max_size ** 2) < self.points_used:
                self._regenerate_cache()

            batch_result.append(mask)

        return np.array(batch_result)


class MNISTMaskGenerator(MixtureMaskGenerator):
    def __init__(self, dim=28, **kwargs):
        half_dim = dim // 2
        generators = [
            ImageBernoulliMaskGenerator(0.5),
            FixedRectangleMaskGenerator(0, 0, dim, half_dim),
            FixedRectangleMaskGenerator(0, 0, half_dim, dim),
            FixedRectangleMaskGenerator(0, half_dim, dim, dim),
            FixedRectangleMaskGenerator(half_dim, 0, dim, dim),
            SquareMaskGenerator(half_dim),
            RectangleMaskGenerator(),
        ]
        weights = [2, 1, 1, 1, 1, 2, 2]

        super().__init__(generators, weights=weights, **kwargs)


class OmniglotMaskGenerator(MixtureMaskGenerator):
    def __init__(self, **kwargs):
        dim = 28
        half_dim = dim // 2
        generators = [
            ImageBernoulliMaskGenerator(0.5),
            FixedRectangleMaskGenerator(0, 0, dim, half_dim),
            FixedRectangleMaskGenerator(0, 0, half_dim, dim),
            FixedRectangleMaskGenerator(0, half_dim, dim, dim),
            FixedRectangleMaskGenerator(half_dim, 0, dim, dim),
            SquareMaskGenerator(half_dim),
            RectangleMaskGenerator(0.1, 0.6),
        ]
        weights = [2, 1, 1, 1, 1, 2, 2]

        super().__init__(generators, weights=weights, **kwargs)


class Cifar10MaskGenerator(MixtureMaskGenerator):
    def __init__(self, **kwargs):
        dim = 32
        half_dim = dim // 2
        generators = [
            ImageBernoulliMaskGenerator(0.3),
            FixedRectangleMaskGenerator(0, 0, dim, half_dim),
            FixedRectangleMaskGenerator(0, 0, half_dim, dim),
            FixedRectangleMaskGenerator(0, half_dim, dim, dim),
            FixedRectangleMaskGenerator(half_dim, 0, dim, dim),
            SquareMaskGenerator(half_dim),
            RectangleMaskGenerator(0.1, 0.5),
        ]
        weights = [2, 1, 1, 1, 1, 2, 2]

        super().__init__(generators, weights=weights, **kwargs)


class GCFMaskGenerator(MixtureMaskGenerator):
    def __init__(self):
        generators = [
            FixedRectangleMaskGenerator(26, 17, 58, 36),
            FixedRectangleMaskGenerator(26, 29, 58, 48),
            FixedRectangleMaskGenerator(26, 15, 37, 50),
            FixedRectangleMaskGenerator(26, 15, 37, 34),
            FixedRectangleMaskGenerator(26, 31, 37, 50),
            FixedRectangleMaskGenerator(43, 20, 62, 44),
        ]
        weights = [1] * 6
        super().__init__(generators, weights=weights)


class SIIDGMMaskGenerator(MixtureMaskGenerator):
    def __init__(self):
        generators = [
            RandomPatternMaskGenerator(max_size=10000, resolution=0.06),
            ImageBernoulliMaskGenerator(0.2),
            FixedRectangleMaskGenerator(16, 16, 48, 48),
            FixedRectangleMaskGenerator(0, 0, 64, 32),
            FixedRectangleMaskGenerator(0, 0, 32, 64),
            FixedRectangleMaskGenerator(0, 32, 64, 64),
            FixedRectangleMaskGenerator(32, 0, 64, 64),
        ]
        weights = [2, 2, 2, 1, 1, 1, 1]
        super().__init__(generators, weights=weights)


class CelebAMaskGenerator(MixtureMaskGenerator):
    def __init__(self):
        generators = [
            SIIDGMMaskGenerator(),
            GCFMaskGenerator(),
            RectangleMaskGenerator(),
        ]
        weights = [1, 1, 2]
        super().__init__(generators, weights=weights)


def get_mask_generator(mask_generator_name, **kwargs):
    return {
        "BernoulliMaskGenerator": BernoulliMaskGenerator,
        "UniformMaskGenerator": UniformMaskGenerator,
        "MNISTMaskGenerator": MNISTMaskGenerator,
        "OmniglotMaskGenerator": OmniglotMaskGenerator,
        "CelebAMaskGenerator": CelebAMaskGenerator,
    }[mask_generator_name](**kwargs)


def get_add_mask_fn(mask_generator):
    def fn(d):
        key = "image" if "image" in d else "features"
        x = d[key]
        [mask] = tf.py_function(mask_generator, [tf.shape(x)], [x.dtype])
        if key == "image":
            mask = tf.reshape(mask, [*x.shape[:-1], 1])
        else:
            mask = tf.reshape(mask, x.shape)
        d["mask"] = mask
        return d

    return fn
