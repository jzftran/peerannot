import numpy as np

from peerannot.models.aggregation.diagonal_multinomial_online import (
    DiagonalMultinomialOnline,
)


class FlatDiagonalOnline(DiagonalMultinomialOnline):
    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_tasks, _, n_classes = batch_matrix.shape
        T = np.zeros((n_tasks, n_classes))

        for i in range(n_tasks):
            worker_labels = batch_matrix[i]

            for j in range(n_classes):
                diag_contrib = np.power(
                    batch_pi[:, j],
                    worker_labels[:, j],
                )  # (n_workers,)

                mask = np.ones(n_classes, dtype=bool)
                mask[j] = False

                off_diag_labels = worker_labels[:, mask]

                # each worker gets their own off-diagonal probs reweighted by rho
                off_diag_probs = (
                    (1 - batch_pi[:, j])[:, None] * batch_rho[mask][None, :]
                ) / (1 - batch_rho[j])  # (n_workers, n_classes-1)

                off_diag_contrib = np.prod(
                    off_diag_probs**off_diag_labels,
                    axis=1,
                )  # (n_workers,)

                worker_probs = diag_contrib * off_diag_contrib
                T[i, j] = np.prod(worker_probs) * batch_rho[j]

        denom = T.sum(axis=1, keepdims=True)
        T = np.where(denom > 0, T / denom, T)
        return T, denom
