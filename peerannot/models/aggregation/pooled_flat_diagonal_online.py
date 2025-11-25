# %%
from __future__ import annotations

import numpy as np
import sparse as sp
from line_profiler import profile

from peerannot.models.aggregation.mongo_online_helpers import EStepResult
from peerannot.models.aggregation.pooled_diagonal_multinomial_online import (
    PooledDiagonalMultinomialOnline,
    VectorizedPooledDiagonalMultinomialOnlineMongo,
)


class VectorizedPooledFlatDiagonalOnline(PooledDiagonalMultinomialOnline):
    @profile
    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        worker_labels_sum = batch_matrix.sum(
            axis=1,
        )  # shape (n_task, n_classes)

        diag_contrib_matrix = (
            batch_pi[None, :] ** worker_labels_sum
        )  # shape (n_task, n_classes)

        # Compute off_diag_terms
        term_k = (1 - batch_pi) * batch_rho
        denom = 1 - batch_rho
        off_diag_terms = term_k / denom[:, None]
        np.fill_diagonal(
            off_diag_terms,
            1,
        )  # set diagonal to 1 to not affect the product

        worker_labels_sum_expanded = worker_labels_sum[:, None, :]

        off_diag_terms_expanded = off_diag_terms[None, :, :]

        powered_terms = off_diag_terms_expanded**worker_labels_sum_expanded

        off_diag_contrib_matrix = np.prod(powered_terms, axis=2)

        T = diag_contrib_matrix * off_diag_contrib_matrix

        batch_denom_e_step = T.sum(axis=1, keepdims=True)
        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)
        return batch_T, batch_denom_e_step


class VectorizedPooledFlatDiagonalOnlineMongo(
    VectorizedPooledDiagonalMultinomialOnlineMongo,
):
    @profile
    def _e_step(
        self,
        batch_matrix: sp.COO,
        batch_pi: sp.COO | np.array,
        batch_rho: sp.COO,
    ) -> EStepResult:
        worker_labels_sum = batch_matrix.sum(
            axis=1,
        )  # shape (n_task, n_classes)

        diag_contrib_matrix = (
            batch_pi[None, :] ** worker_labels_sum
        )  # shape (n_task, n_classes)

        term_k = (1 - batch_pi) * batch_rho
        denom = (1 - batch_rho).todense()[:, None]
        off_diag_terms = np.where(
            denom > 0,
            term_k / denom,
            term_k,
        )  # shape (n_classes, n_classes)

        n_classes = off_diag_terms.shape[0]
        identity = sp.eye(n_classes, n_classes, format="coo")

        # Replace diagonal with 1
        off_diag_terms = off_diag_terms * (1 - identity) + identity

        worker_labels_sum_expanded = worker_labels_sum[:, None, :]

        off_diag_terms_expanded = off_diag_terms[None, :, :]

        powered_terms = off_diag_terms_expanded**worker_labels_sum_expanded

        off_diag_contrib_matrix = np.prod(powered_terms, axis=2)

        T = diag_contrib_matrix * off_diag_contrib_matrix

        batch_denom_e_step = T.sum(axis=1, keepdims=True)

        if not np.any(batch_denom_e_step == 0):
            batch_denom_e_step = batch_denom_e_step.todense()
            T.fill_value = 0.0  # dirty solution solving the mixed sparse-dense operation error

        batch_T = np.where(
            batch_denom_e_step > 0,
            T / batch_denom_e_step,
            T,
        )

        return EStepResult(batch_T, batch_denom_e_step)
