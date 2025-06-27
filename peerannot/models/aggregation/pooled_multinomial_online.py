from typing import Annotated

import numpy as np
from annotated_types import Ge, Gt
from pydantic import validate_call

from peerannot.models.aggregation.online_helpers import OnlineAlgorithm
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


class PooledMultinomialOnline(OnlineAlgorithm):
    @validate_call
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
    ) -> None:
        super().__init__(gamma0, decay)

    def _expand_pi(self, new_n_workers: int, new_n_classes: int) -> None:
        if new_n_classes > self.n_classes:
            self.pi = self._expand_array(
                self.pi,
                (new_n_classes, new_n_classes),
                fill_value=0.0,
            )

    def _initialize_pi(self) -> None:
        self.pi = np.zeros((self.n_classes, self.n_classes))

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)

        aggregated_votes = np.einsum(
            "tq, tij -> qj",
            batch_T,
            batch_matrix,
        )  # shape (n_classes, n_classes)
        denom = aggregated_votes.sum(axis=1, keepdims=True)

        batch_pi = np.divide(
            aggregated_votes,
            denom,
            out=np.zeros_like(aggregated_votes),
            where=denom != 0,
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

        # use mask instead of power
        for i in range(batch_n_tasks):
            for j in range(batch_n_classes):
                num = (
                    np.prod(
                        np.power(
                            local_pi[j, :],
                            batch_matrix[i, :, :],
                        ),
                    )
                    * local_rho[j]
                )
                T[i, j] = num

        batch_denom_e_step = T.sum(axis=1, keepdims=True)
        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)
        return batch_T, batch_denom_e_step

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ):
        batch_to_global = {
            batch_class_idx: self.class_mapping[class_name]
            for class_name, batch_class_idx in class_mapping.items()
        }

        for i_batch, i_global in batch_to_global.items():
            for j_batch, j_global in batch_to_global.items():
                self.pi[i_global, j_global] = (1 - self.gamma) * self.pi[
                    i_global,
                    j_global,
                ] + self.gamma * batch_pi[
                    i_batch,
                    j_batch,
                ]

            row_sum = self.pi[i_global, :].sum()
            if row_sum > 0:
                self.pi[i_global, :] /= row_sum
