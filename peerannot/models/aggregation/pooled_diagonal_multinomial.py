import numpy as np

from .dawid_skene import DawidSkene


class PooledDiagonalMultinomial(DawidSkene):
    def _m_step(self) -> None:
        """Maximizing log likelihood with only diagonal elements of pi."""
        self.rho = self.T.sum(0) / self.n_task

        diag_votes = np.einsum("tq, tiq -> q", self.T, self.crowd_matrix)
        denom = np.einsum("tq, tij -> q", self.T, self.crowd_matrix)

        self.pi = np.divide(
            diag_votes,
            denom,
            out=np.zeros_like(diag_votes),
            where=denom != 0,
        )

        self.pi_non_diag_values = (np.ones_like(self.pi) - self.pi) / (
            self.n_classes - 1
        )

    def _e_step(self) -> None:
        """Estimate indicator variables (see eq. 2.5 Dawid and Skene 1979)

        Returns:
            T: New estimate for indicator variables (n_task, n_worker)
            denom: value used to compute likelihood easily
        """
        T = np.zeros((self.n_task, self.n_classes))

        for i in range(self.n_task):
            for j in range(self.n_classes):
                worker_labels = self.crowd_matrix[i]
                diag_contrib = np.prod(np.power(self.pi, worker_labels))
                mask = np.ones(self.n_classes, dtype=bool)
                mask[j] = False
                off_diag_contrib = np.prod(
                    np.power(
                        self.pi_non_diag_values[mask],
                        worker_labels[:, mask],
                    ),
                )

                T[i, j] = diag_contrib * off_diag_contrib * self.rho[j]

        self.denom_e_step = T.sum(1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
