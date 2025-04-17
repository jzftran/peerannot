import numpy as np

from .dawid_skene import DawidSkene


class DiagonalMultinomial(DawidSkene):
    def _m_step(self) -> None:
        """Maximizing log likelihood with only diagonal elements of pi."""
        rho = self.T.sum(0) / self.n_task

        pi = np.zeros((self.n_workers, self.n_classes))
        for j in range(self.n_classes):
            # TODO @jzftran: change this to calculate diagonal faster
            pij = self.T[:, j] @ self.crowd_matrix.transpose((1, 0, 2))
            denom = pij.sum(1)

            diag_values = pij[:, j] / np.where(denom > 0, denom, 1e-9)
            pi[:, j] = diag_values

        # pi shape (n_workers, n_class), reresents how sure worker is
        # sure that the label j is true
        pi_non_diag_values = (np.ones_like(pi) - pi) / (self.n_classes - 1)
        self.rho, self.pi, self.pi_non_diag_values = (
            rho,
            pi,
            pi_non_diag_values,
        )

    def _e_step(self) -> None:
        T = np.zeros((self.n_task, self.n_classes))

        for i in range(self.n_task):
            for j in range(self.n_classes):
                # Get all worker labels for task i (n_workers, n_classes)
                worker_labels = self.crowd_matrix[
                    i
                ]  # shape (n_workers, n_classes)

                # Vectorized computation for all workers simultaneously
                # Diagonal contributions: pi[k,j]^worker_labels[k,j]
                diag_contrib = np.power(
                    self.pi[:, j],
                    worker_labels[:, j],
                )  # shape (n_workers,)

                # Off-diagonal contributions: product over l≠j of
                # pi_non_diag[k,j]^worker_labels[k,l]
                mask = np.ones(self.n_classes, dtype=bool)
                mask[j] = False  # exclude current class j
                off_diag_labels = worker_labels[
                    :,
                    mask,
                ]  # shape (n_workers, n_classes-1)

                off_diag_contrib = np.prod(
                    np.power(
                        self.pi_non_diag_values[:, j][:, np.newaxis],
                        off_diag_labels,
                    ),
                    axis=1,
                )  # shape (n_workers,)

                worker_probs = (
                    diag_contrib * off_diag_contrib
                )  # shape (n_workers,)
                T[i, j] = np.prod(worker_probs) * self.rho[j]

        self.denom_e_step = T.sum(1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)


class VectorizedDiagonalMultinomial(DiagonalMultinomial):
    def _m_step(self) -> None:
        """Maximizing log likelihood with only diagonal elements of pi."""
        rho = self.T.sum(0) / self.n_task
        pij = np.einsum("tq,tiq->iq", self.T, self.crowd_matrix)
        denom = np.einsum("tq, tij -> iq", self.T, self.crowd_matrix)
        pi = np.where(denom > 0, pij / denom, 0)
        # pi shape (n_workers, n_classes)
        pi_non_diag_values = (np.ones_like(pi) - pi) / (self.n_classes - 1)

        self.rho, self.pi, self.pi_non_diag_values = (
            rho,
            pi,
            pi_non_diag_values,
        )

    def _e_step(self) -> None:
        """Vectorized implementation of e-step without worker loops."""
        # Compute diagonal contributions
        # shape: (n_task, n_workers, n_classes)
        diag_contrib = np.power(
            self.pi[np.newaxis, :, :],  # (1, n_workers, n_classes)
            self.crowd_matrix,  # (n_task, n_workers, n_classes)
        )

        # Compute off-diagonal contributions
        # For each class j, we need to multiply pi_non_diag_values[k,j]^worker_labels[l] for all l != j
        mask = 1 - np.eye(self.n_classes)  # (n_casses, n_classes)

        # shape: (n_task, n_workers, n_classes, n_classes)
        off_diag_powers = np.power(
            self.pi_non_diag_values[
                np.newaxis,
                :,
                np.newaxis,
                :,
            ],  # (1, n_workers, 1, n_classes)
            self.crowd_matrix[:, :, :, np.newaxis]
            * mask[np.newaxis, np.newaxis, :, :],
        )

        off_diag_contrib = np.prod(
            off_diag_powers,
            axis=2,
        )  # (n_task, n_workers, n_classes)

        worker_probs = diag_contrib * off_diag_contrib

        T = (
            np.prod(worker_probs, axis=1) * self.rho[np.newaxis, :]
        )  # (n_task, n_classes)

        self.denom_e_step = T.sum(axis=1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
