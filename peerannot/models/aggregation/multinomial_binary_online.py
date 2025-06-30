from typing import Annotated

import numpy as np
from annotated_types import Ge, Gt
from pydantic import validate_call

from peerannot.models.aggregation.online_helpers import OnlineAlgorithm
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


class MultinomialBinaryOnline(OnlineAlgorithm):
    @validate_call
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
    ) -> None:
        super().__init__(gamma0, decay)

    def _expand_pi(self, new_n_workers: int, new_n_classes: int) -> None:
        """Expand the pi array if the number of workers or classes increases."""
        if new_n_workers > self.n_workers:
            self.pi = self._expand_array(
                self.pi,
                (new_n_workers, 1),
                fill_value=0.0,
            )

    def _initialize_pi(self) -> None:
        self.pi = np.zeros((self.n_workers, 1))

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
        # Update only workers present in the batch
        for worker, batch_worker_idx in worker_mapping.items():
            worker_idx = self.worker_mapping[worker]
            self.pi[worker_idx] = (1 - self.gamma) * self.pi[
                worker_idx,
            ] + self.gamma * batch_pi[batch_worker_idx,]

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)
        batch_n_workers = batch_matrix.shape[1]

        batch_pi = np.zeros(batch_n_workers)
        for j in range(batch_n_workers):
            labeled_count = batch_matrix[:, j, :].sum()
            weighted_sum = (batch_T * batch_matrix[:, j, :]).sum()
            alpha = np.divide(
                weighted_sum,
                labeled_count,
                out=np.zeros_like(weighted_sum),
                where=labeled_count > 0,
            )
            batch_pi[j] = alpha

        return batch_rho, batch_pi

    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_n_tasks = batch_matrix.shape[0]
        batch_n_classes = batch_matrix.shape[2]

        off_diag_alpha = (np.ones_like(batch_pi) - batch_pi) / (
            batch_n_classes - 1
        )
        T = np.zeros((batch_n_tasks, batch_n_classes))
        for i in range(batch_n_tasks):
            for j in range(batch_n_classes):
                diag_contrib = np.prod(
                    np.power(
                        batch_pi,
                        batch_matrix[i, :, j],
                    ),
                )  # shape (n_workers,)

                mask = np.ones(batch_n_classes, dtype=bool)
                mask[j] = False
                off_diag_labels = batch_matrix[i, :, mask]

                off_diag_contrib = np.prod(
                    np.power(off_diag_alpha, off_diag_labels),
                )

                T[i, j] = (
                    np.prod(diag_contrib * off_diag_contrib) * batch_rho[j]
                )

        batch_denom_e_step = T.sum(1, keepdims=True)

        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)
        return batch_T, batch_denom_e_step
