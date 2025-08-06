from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
)

import numpy as np
from annotated_types import Ge, Gt
from pydantic import validate_call

from peerannot.models.aggregation.mongo_online_helpers import (
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
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
    ) -> None:
        super().__init__(gamma0, decay)

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
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
    ) -> None:
        super().__init__(gamma0, decay)

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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

        return batch_rho, batch_pi

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ):
        doc = self.db.worker_confusion_matrices.find_one(
            {"_id": "pooledMultinomialBinary"},
        )
        pi_old = (
            np.array(doc["confusion_matrix"])
            if doc
            else np.full_like(batch_pi, 0.5)  # init pi as 0.5
        )

        pi_new = pi_old + self.gamma * (batch_pi - pi_old)

        # Store updated pi
        self.db.worker_confusion_matrices.update_one(
            {"_id": "pooledMultinomialBinary"},
            {"$set": {"confusion_matrix": pi_new.tolist()}},
            upsert=True,
        )

    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ):
        batch_n_classes = batch_matrix.shape[2]

        batch_n_il = np.sum(batch_matrix, axis=1)

        n_i = batch_n_il.sum(axis=1, keepdims=True)

        diag_contrib = batch_pi**batch_n_il
        off_diag_factor = (1 - batch_pi) / (batch_n_classes - 1)
        off_diag_contrib = off_diag_factor ** (n_i - batch_n_il)

        T = diag_contrib * off_diag_contrib * batch_rho[np.newaxis, :]

        batch_denom_e_step = T.sum(axis=1, keepdims=True)

        batch_denom_e_step = batch_denom_e_step.todense()

        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)
        return batch_T, batch_denom_e_step
