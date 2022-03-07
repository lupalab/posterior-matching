from typing import Callable, Dict, Any

import haiku as hk
import jax
import numpy as np
from bax import TrainState
from bax.callbacks import Callback
from chex import ArrayTree, Array, PRNGKey
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def _make_cost_m(cm):
    s = np.max(cm)
    return -cm + s


def clustering_accuracy(y_true: Array, y_pred: Array) -> float:
    """Computes the "clustering accuracy" metric.

    This is defined as the maximum obtainable accuracy when considering all possible
    assignments of clusters to class labels. Rather than trying all possible
    permutations, this problem is solved efficiently by formulating it as the linear
    sum assignment problem.

    Args:
        y_true: The ground truth labels.
        y_pred: The cluster assignments.

    Returns:
        The clustering accuracy.
    """
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(_make_cost_m(cm))
    total = cm[row_ind, col_ind].sum()

    return total * 1.0 / np.sum(cm)


class ClusteringAccuracyCallback(Callback):
    """Callback that computes the clustering accuracy metric.

    Args:
        pred_fn: A function that takes data batches as input and returns the model's
            cluster assignments.
    """

    def __init__(self, pred_fn: Callable[[ArrayTree], Array]):
        self._pred_fn = jax.jit(hk.transform_with_state(pred_fn).apply)
        self._preds = []
        self._labels = []

    def on_validation_step(
        self, train_state: TrainState, key: PRNGKey, batch: ArrayTree
    ):
        preds, _ = self._pred_fn(train_state.params, train_state.state, key, batch)

        self._labels.append(batch["label"])
        self._preds.append(preds)

    def on_validation_end(
        self, train_state: TrainState, step: int, logs: Dict[str, Any]
    ):
        y_true = np.hstack(self._labels)
        y_pred = np.hstack(self._preds)

        acc = clustering_accuracy(y_true, y_pred)

        logs["val_clustering_accuracy"] = acc

        self._labels.clear()
        self._preds.clear()
