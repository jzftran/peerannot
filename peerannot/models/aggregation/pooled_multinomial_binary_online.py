from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Annotated,
)

import numpy as np
from annotated_types import Ge, Gt
from pydantic import validate_call

from peerannot.models.aggregation.dawid_skene_online import OnlineAlgorithm

if TYPE_CHECKING:
    from peerannot.models.aggregation.types import (
        ClassMapping,
        WorkerMapping,
    )


class PoooledMultinomialBinaryOnline(OnlineAlgorithm):
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
        local_pi: np.ndarray,
        local_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_n_tasks = batch_matrix.shape[0]
        batch_n_classes = batch_matrix.shape[2]

        T = np.zeros((batch_n_tasks, batch_n_classes))

        for i in range(batch_n_tasks):
            batch_n_il = np.sum(
                batch_matrix,
                axis=1,
            )
            batch_n_i = batch_n_il[
                i
            ].sum()  # total numer of annotators of task i

            n_i = batch_n_i

            for l in range(batch_n_classes):
                n_il = batch_n_il[
                    i,
                    l,
                ]  # numer of annotators of task i voting for label l
                diag_contrib = np.power(local_pi, n_il)

                denominator = batch_n_classes - 1
                off_diag_contrib = np.power(
                    np.divide(
                        1 - local_pi,
                        denominator,
                        out=np.zeros_like(local_pi),
                        where=(denominator != 0),
                    ),
                    n_i - n_il,
                )

                T[i, l] = diag_contrib * off_diag_contrib * local_rho[l]

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
