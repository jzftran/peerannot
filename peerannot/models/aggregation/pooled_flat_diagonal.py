import numpy as np

from peerannot.models.aggregation.pooled_diagonal_multinomial import (
    PooledDiagonalMultinomial,
)


class PooledFlatDiagonal(PooledDiagonalMultinomial):
    """iNaturalist flavour diagonal model"""

    def _e_step(self) -> None:
        T = np.zeros((self.n_task, self.n_classes))
        for i in range(self.n_task):
            for j in range(self.n_classes):
                worker_labels = self.crowd_matrix[i]
                diag_contrib = np.prod(
                    np.power(self.pi[j], worker_labels[:, j]),
                )
                mask = np.ones(self.n_classes, dtype=bool)
                mask[j] = False
                off_diag_contrib = np.prod(
                    np.power(
                        ((1 - self.pi)[mask] * self.rho[mask])
                        / (1 - self.rho[j]),
                        worker_labels[:, mask].sum(
                            axis=0,
                        ),
                    ),
                )
                T[i, j] = diag_contrib * off_diag_contrib
        self.denom_e_step = T.sum(1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
