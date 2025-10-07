from __future__ import annotations

import numpy as np
import sparse as sp

from peerannot.models.aggregation.mongo_online_helpers import EStepResult
from peerannot.models.aggregation.multinomial_binary_online import (
    MultinomialBinaryOnline,
    VectorizedMultinomialBinaryOnlineMongo,
)


class FlatSingleBinomialOnline(MultinomialBinaryOnline):
    def _e_step(self, batch_matrix, batch_pi, batch_rho):
        n_tasks, n_workers, n_classes = batch_matrix.shape

        T = np.zeros((n_tasks, n_classes))
        for i in range(n_tasks):
            for j in range(n_classes):
                # Shape: (n_workers, n_classes)
                conf = ((1 - batch_pi)[:, None] * batch_rho[None, :]) / (
                    1 - batch_rho[j]
                )
                conf[:, j] = batch_pi  # replace diagonal with pi_j

                # Votes of all workers (shape: n_workers, n_classes)
                votes = batch_matrix[i]

                # Contribution from all workers for candidate j
                contrib = np.prod(
                    conf**votes,
                    axis=(0, 1),
                )

                T[i, j] = contrib * batch_rho[j]

        denom = T.sum(axis=1, keepdims=True)
        T = np.where(denom > 0, T / denom, T)
        return T, denom


# %%
class VectorizedFlatSingleBinomialOnline(MultinomialBinaryOnline):
    def _e_step(self, batch_matrix, batch_pi, batch_rho):
        n_tasks, n_workers, n_classes = batch_matrix.shape

        # Off-diagonal part:
        # (n_classes, n_workers, n_classes)
        off_diag = (
            (1 - batch_pi)[None, :, None] * batch_rho[None, None, :]
        ) / (1 - batch_rho)[:, None, None]

        # Identity mask for diagonals: (n_classes, n_classes)
        eye = np.eye(n_classes, dtype=bool)

        # Broadcast to (n_classes, n_workers, n_classes)
        conf = np.where(eye[:, None, :], batch_pi[None, :, None], off_diag)

        # Add candidate axis for votes
        votes = batch_matrix[
            :,
            None,
            :,
            :,
        ]  # (n_tasks, 1, n_workers, n_classes)

        # Expand conf to match
        conf = conf[None, :, :, :]  # (1, n_classes, n_workers, n_classes)

        # Multiply contributions
        contrib = np.prod(conf**votes, axis=(2, 3))  # (n_tasks, n_classes)

        # Apply priors
        T = contrib * batch_rho[None, :]

        denom = T.sum(axis=1, keepdims=True)
        T = np.where(denom > 0, T / denom, T)
        return T, denom

    @property
    def pi(self):
        raise NotImplementedError


class VectorizedFlatSingleBinomialOnlineMongo(
    VectorizedMultinomialBinaryOnlineMongo,
):
    def _e_step(
        self,
        batch_matrix: sp.COO,
        batch_pi: sp.COO | np.array,
        batch_rho: sp.COO,
    ) -> EStepResult:
        n_tasks, _, n_classes = batch_matrix.shape
        tasks, workers, assigned_classes = batch_matrix.coords
        counts = batch_matrix.data  # shape: (nnz,)

        one_minus_pi = 1.0 - batch_pi[workers]  # shape: (nnz,)
        rho_c = batch_rho[assigned_classes]  # shape: (nnz,)
        numerator = one_minus_pi * rho_c  # shape: (nnz,)
        off_diag = numerator[:, None] / (
            1.0 - batch_rho[None, :]
        )  # shape: (nnz, n_classes)

        diag_mask = (
            assigned_classes[:, None] == np.arange(n_classes)[None, :]
        )  # shape: (nnz, n_classes)
        pi_wk_expanded = np.tile(
            batch_pi[workers][:, None].todense(),
            (1, n_classes),
        )  # shape: (nnz, n_classes)

        conf = np.where(diag_mask, pi_wk_expanded, off_diag)

        term = conf ** counts[:, None]  # shape: (nnz, n_classes)

        product = np.ones((n_tasks, n_classes), dtype=np.float64)
        np.multiply.at(
            product,
            (tasks[:, None], np.arange(n_classes)[None, :]),
            term,
        )

        T = product * batch_rho[None, :]

        denom = T.sum(axis=1, keepdims=True).todense()
        batch_T = np.where(denom > 0, T / denom, T)
        return EStepResult(batch_T, denom)
