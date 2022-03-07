import numpy as np
import tensorflow_hub as hub
from tqdm import tqdm


def get_inception_embeddings(images, batch_size=32, verbose=True):
    inception = hub.load("https://tfhub.dev/tensorflow/tfgan/eval/inception/1")

    if images.shape[-1] == 1:
        images = np.tile(images, [1, 1, 1, 3])

    images = (images * 255).astype(np.uint8)

    out = []

    for i in tqdm(
        range(0, len(images), batch_size),
        desc="Generating Embeddings",
        disable=not verbose,
    ):
        emb = inception(images[i : i + batch_size])
        out.append(np.reshape(emb["pool_3"], [-1, 2048]))

    return np.concatenate(out, axis=0)
