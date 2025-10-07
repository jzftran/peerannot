from __future__ import annotations

from typing import (
    TYPE_CHECKING,
)

import numpy as np
from line_profiler import profile
from pydantic import validate_call

from peerannot.models.aggregation.mongo_online_helpers import (
    EStepResult,
    MStepResult,
    SparseMongoOnlineAlgorithm,
)
from peerannot.models.aggregation.online_helpers import (
    OnlineAlgorithm,
)

if TYPE_CHECKING:
    from peerannot.models.aggregation.types import (
        ClassMapping,
        WorkerMapping,
    )
import sparse as sp


class PooledMultinomialBinaryOnline(OnlineAlgorithm):
    @validate_call
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.pi = np.array(0)  # 1 elem np.array

    def _expand_pi(self, new_n_workers: int, new_n_classes: int) -> None:
        # no need to expand_pi
        pass

    def _initialize_pi(self) -> None:
        self.pi = np.array(0.5)

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)

        sum_diag_votes = np.einsum(
            "tq, tiq ->",
            batch_T,
            batch_matrix,
        )  # trace(T.T @ crowd_matrix)

        batch_total_votes = batch_matrix.sum()
        batch_pi = np.divide(
            sum_diag_votes,
            batch_total_votes,
            out=np.zeros_like(sum_diag_votes),
            where=(batch_total_votes != 0),
        )

        return batch_rho, batch_pi

    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_n_tasks = batch_matrix.shape[0]
        batch_n_classes = batch_matrix.shape[2]

        T = np.zeros((batch_n_tasks, batch_n_classes))

        batch_n_il = np.sum(
            batch_matrix,
            axis=1,
        )
        for i in range(batch_n_tasks):
            batch_n_i = batch_n_il[
                i
            ].sum()  # total numer of annotators of task i

            for l in range(batch_n_classes):
                diag_contrib = np.power(batch_pi, batch_n_il[i, l])

                denominator = batch_n_classes - 1
                off_diag_contrib = np.power(
                    np.divide(
                        1 - batch_pi,
                        denominator,
                        out=np.zeros_like(batch_pi),
                        where=(denominator != 0),
                    ),
                    batch_n_i - batch_n_il[i, l],
                )

                T[i, l] = diag_contrib * off_diag_contrib * batch_rho[l]

        batch_denom_e_step = T.sum(axis=1, keepdims=True)

        batch_T = np.where(
            batch_denom_e_step > 0,
            np.divide(T, batch_denom_e_step),
            T,
        )

        return batch_T, batch_denom_e_step

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ):
        self.pi = self.pi + self.gamma * (batch_pi - self.pi)


class VectorizedPooledMultinomialBinaryOnlineMongo(SparseMongoOnlineAlgorithm):
    """Vectorized pooled multinomial binary online algorithm using sparse matrices and mongo."""

    @validate_call
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @profile
    def _m_step(
        self,
        batch_matrix: sp.COO,
        batch_T: sp.COO,
    ) -> MStepResult:
        batch_rho = batch_T.mean(axis=0)

        batch_T = sp.COO(batch_T)

        # sp.einsum behaves differently
        sum_diag_votes = np.einsum(
            "tq, tiq ->",
            batch_T,
            batch_matrix,
        )  # trace(T.T @ crowd_matrix)

        batch_total_votes = batch_matrix.sum()
        batch_pi = np.divide(
            sum_diag_votes,
            batch_total_votes,
            out=np.zeros_like(sum_diag_votes),
            where=(batch_total_votes != 0),
        )

        return MStepResult(batch_rho, batch_pi)

    @profile
    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ):
        doc = self.db.worker_confusion_matrices.find_one(
            {"_id": self.__class__.__name__},
        )
        pi_old = (
            np.array(doc["confusion_matrix"])
            if doc
            else np.full_like(batch_pi, 0.5)  # init pi as 0.5
        )

        pi_new = pi_old + self.gamma * (batch_pi - pi_old)

        # Store updated pi
        self.db.worker_confusion_matrices.update_one(
            {"_id": self.__class__.__name__},
            {"$set": {"confusion_matrix": pi_new.tolist()}},
            upsert=True,
        )

    @property
    def pi(self) -> np.ndarray:
        doc = self.db.worker_confusion_matrices.find_one(
            {"_id": self.__class__.__name__},
        )
        if doc is not None and "confusion_matrix" in doc:
            return np.array(doc["confusion_matrix"])

    @profile
    def _e_step(
        self,
        batch_matrix: sp.COO,
        batch_pi: sp.COO | np.array,
        batch_rho: sp.COO,
    ) -> EStepResult:
        n_tasks, _, n_classes = batch_matrix.shape

        # Extract COO coords & counts from batch_matrix
        tasks, _, assigned_classes = batch_matrix.coords
        counts = batch_matrix.data

        # Create match mask (nnz, n_classes)
        # This mask is True when the assigned class matches each possible true class
        match_mask = assigned_classes[:, None] == np.arange(n_classes)[None, :]

        # Compute per-worker off-diagonal probabilities
        # Since batch_pi is scalar, off_diag_alpha is the same for all workers
        off_diag_alpha = (1.0 - batch_pi) / (n_classes - 1)

        # Compute probabilities for each non-zero entry and each possible true class
        probs_nnz = (
            np.where(
                match_mask,
                batch_pi,  # Correct assignment probability (scalar)
                off_diag_alpha,  # Incorrect assignment probability (scalar)
            )
            ** counts[:, None]  # Raise to power of counts
        )

        # Initialize likelihood matrix
        likelihood = np.ones((n_tasks, n_classes), dtype=np.float64)

        # Accumulate products into the likelihood matrix
        # For each non-zero entry, multiply its contribution into the likelihood
        # for its task and all possible classes
        np.multiply.at(
            likelihood,
            (
                tasks[:, None],
                np.arange(n_classes)[None, :],
            ),  # Indices to update
            probs_nnz,  # Values to multiply in
        )

        # Compute T by multiplying likelihood by class priors
        T = likelihood * batch_rho[None, :]

        # Compute normalization constants and final probabilities
        denom = T.sum(axis=1, keepdims=True).todense()
        batch_T = np.where(denom > 0, T / denom, T)

        return EStepResult(batch_T, denom)

    def build_full_pi_tensor(self) -> np.ndarray:
        pi = self.pi
        n_classes = self.n_classes
        n_workers = self.n_workers

        # Off-diagonal probability for each worker and class (row-based)
        off_diag = (1.0 - pi) / (n_classes - 1)  # (n_workers, n_classes)

        # Broadcast  full_pi[w, i, j] = off_diag[w, i] for all j
        full_pi = np.broadcast_to(
            off_diag,
            (n_workers, n_classes, n_classes),
        ).copy()

        # Replace diagonals with pi
        idx = np.arange(n_classes)
        full_pi[:, idx, idx] = pi

        return full_pi
