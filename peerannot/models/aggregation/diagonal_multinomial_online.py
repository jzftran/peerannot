from typing import Annotated

import numpy as np
from annotated_types import Ge, Gt
from pydantic import validate_call

from peerannot.models.aggregation.online_helpers import OnlineAlgorithm
from peerannot.models.aggregation.types import ClassMapping, WorkerMapping


class DiagonalMultinomialOnline(OnlineAlgorithm):
    @validate_call
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
    ) -> None:
        super().__init__(gamma0, decay)

    def _expand_pi(self, new_n_workers: int, new_n_classes: int) -> None:
        """Expand the pi array if the number of workers or classes increases."""
        if new_n_workers > self.n_workers or new_n_classes > self.n_classes:
            self.pi = self._expand_array(
                self.pi,
                (new_n_workers, new_n_classes),
                fill_value=0.0,
            )

    def _initialize_pi(self) -> None:
        self.pi = np.zeros((self.n_workers, self.n_classes))

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
        # Update only workers present in the batch
        for worker, batch_worker_idx in worker_mapping.items():
            worker_idx = self.worker_mapping[worker]

            for i_batch, i_global in batch_to_global.items():
                # for j_batch, j_global in batch_to_global.items():
                self.pi[worker_idx, i_global] = (1 - self.gamma) * self.pi[
                    worker_idx,
                    i_global,
                ] + self.gamma * batch_pi[
                    batch_worker_idx,
                    i_batch,
                ]

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_rho = batch_T.mean(axis=0)

        batch_n_workers = batch_matrix.shape[1]
        batch_n_classes = batch_matrix.shape[2]

        batch_pi = np.zeros((batch_n_workers, batch_n_classes))

        for j in range(batch_n_classes):
            # TODO @jzftran: change this to calculate diagonal faster
            pij = batch_T[:, j] @ batch_matrix.transpose((1, 0, 2))
            denom = pij.sum(1)

            diag_values = pij[:, j] / np.where(denom > 0, denom, 1e-9)
            batch_pi[:, j] = diag_values

        # pi shape (n_workers, n_class), reresents how sure worker is
        # sure that the label j is true

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

        batch_pi_non_diag_values = (np.ones_like(batch_pi) - batch_pi) / (
            batch_n_classes - 1
        )

        for i in range(batch_n_tasks):
            for j in range(batch_n_classes):
                # Get all worker labels for task i (n_workers, n_classes)
                worker_labels = batch_matrix[i]  # shape (n_workers, n_classes)

                # Vectorized computation for all workers simultaneously
                # Diagonal contributions: pi[k,j]^worker_labels[k,j]
                diag_contrib = np.power(
                    batch_pi[:, j],
                    worker_labels[:, j],
                )  # shape (n_workers,)

                # Off-diagonal contributions: product over l≠j of
                # pi_non_diag[k,j]^worker_labels[k,l]
                mask = np.ones(batch_n_classes, dtype=bool)
                mask[j] = False  # exclude current class j
                off_diag_labels = worker_labels[
                    :,
                    mask,
                ]  # shape (n_workers, n_classes-1)

                off_diag_contrib = np.prod(
                    np.power(
                        batch_pi_non_diag_values[:, j][:, np.newaxis],
                        off_diag_labels,
                    ),
                    axis=1,
                )  # shape (n_workers,)

                worker_probs = (
                    diag_contrib * off_diag_contrib
                )  # shape (n_workers,)
                T[i, j] = np.prod(worker_probs) * batch_rho[j]

        batch_denom_e_step = T.sum(1, keepdims=True)
        batch_T = np.where(batch_denom_e_step > 0, T / batch_denom_e_step, T)

        return batch_T, batch_denom_e_step
