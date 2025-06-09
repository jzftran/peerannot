import numpy as np

from peerannot.models import DawidSkene


class MultinomialBinary(DawidSkene):
    def _m_step(self) -> None:
        """Maximizing log likelihood with only diagonal elements of pi."""
        rho = self.T.sum(0) / self.n_task

        pi = np.zeros(self.n_workers)
        for j in range(self.n_workers):
            labeled_count = self.crowd_matrix[:, j, :].sum()
            weighted_sum = (self.T * self.crowd_matrix[:, j, :]).sum()
            alpha = np.divide(
                weighted_sum,
                labeled_count,
                out=np.zeros_like(weighted_sum),
                where=labeled_count > 0,
            )
            pi[j] = alpha

        off_diag_alpha = (np.ones_like(pi) - pi) / (self.n_classes - 1)
        self.rho, self.pi, self.off_diag_alpha = rho, pi, off_diag_alpha

    def _e_step(self) -> None:
        """Estimate indicator variables (see eq. 2.5 Dawid and Skene 1979)

        Returns:
            T: New estimate for indicator variables (n_task, n_worker)
            denom: value used to compute likelihood easily
        """
        T = np.zeros((self.n_task, self.n_classes))
        for i in range(self.n_task):
            for j in range(self.n_classes):
                diag_contrib = np.prod(
                    np.power(
                        self.pi,
                        self.crowd_matrix[i, :, j],
                    ),
                )  # shape (n_workers,)

                mask = np.ones(self.n_classes, dtype=bool)
                mask[j] = False
                off_diag_labels = self.crowd_matrix[i, :, mask]

                off_diag_contrib = np.prod(
                    np.power(self.off_diag_alpha, off_diag_labels),
                )

                T[i, j] = (
                    np.prod(diag_contrib * off_diag_contrib) * self.rho[j]
                )

        self.denom_e_step = T.sum(1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
