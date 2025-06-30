from typing import Annotated

import numpy as np
from annotated_types import Ge, Gt
from pydantic import validate_call

from peerannot.models.aggregation.online_helpers import OnlineAlgorithm
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


class PooledDiagonalMultinomialOnline(OnlineAlgorithm):
    @validate_call
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
    ) -> None:
        super().__init__(gamma0, decay)

    def _expand_pi(self, new_n_workers: int, new_n_classes: int) -> None:
        """Expand the pi array if the number of workers or classes increases."""
        if new_n_classes > self.n_classes:
            self.pi = self._expand_array(
                self.pi,
                (new_n_classes,),
                fill_value=0.0,
            )

    def _initialize_pi(self) -> None:
        self.pi = np.zeros(self.n_classes)

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        # For each class in the batch, map batch class idx to global class idx
        batch_to_global = {
            batch_class_idx: self.class_mapping[class_name]
            for class_name, batch_class_idx in class_mapping.items()
        }
        for i_batch, i_global in batch_to_global.items():
            # for j_batch, j_global in batch_to_global.items():
            self.pi[i_global] = (1 - self.gamma) * self.pi[
                i_global
            ] + self.gamma * batch_pi[i_batch]

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)

        diag_votes = np.einsum("tq, tiq -> q", batch_T, batch_matrix)
        denom = np.einsum("tq, tij -> q", batch_T, batch_matrix)

        batch_pi = np.divide(
            diag_votes,
            denom,
            out=np.zeros_like(diag_votes),
            where=denom != 0,
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

        batch_pi_non_diag_values = (np.ones_like(batch_pi) - batch_pi) / (
            batch_n_tasks
        )

        T = np.zeros((batch_n_tasks, batch_n_classes))

        for i in range(batch_n_tasks):
            for j in range(batch_n_classes):
                worker_labels = batch_matrix[i]
                diag_contrib = np.prod(np.power(batch_pi, worker_labels))
                mask = np.ones(batch_n_classes, dtype=bool)
                mask[j] = False
                off_diag_contrib = np.prod(
                    np.power(
                        batch_pi_non_diag_values[mask],
                        worker_labels[:, mask],
                    ),
                )

                T[i, j] = diag_contrib * off_diag_contrib * batch_rho[j]

        batch_denom_e_step = T.sum(1, keepdims=True)
        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)

        return batch_T, batch_denom_e_step
