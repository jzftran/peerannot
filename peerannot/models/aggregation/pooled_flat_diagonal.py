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


class VectorizedPooledFlatDiagonal(PooledDiagonalMultinomial):
    """iNaturalist flavour diagonal model, e_step vectorized"""

    def _e_step(self) -> None:
        worker_labels_sum = self.crowd_matrix.sum(
            axis=1,
        )  # shape (n_task, n_classes)

        # diag_contrib_matrix: self.pi[j] ** worker_labels_sum[i, j] for all i, j
        diag_contrib_matrix = (
            self.pi[None, :] ** worker_labels_sum
        )  # shape (n_task, n_classes)

        # off_diag_terms
        term_k = (1 - self.pi) * self.rho
        denom = 1 - self.rho
        off_diag_terms = (
            term_k / denom[:, None]
        )  # shape (n_classes, n_classes)
        np.fill_diagonal(
            off_diag_terms,
            1,
        )  # set diagonal to 1 to not affect the product

        # worker_labels_sum: (n_task, n_classes) -> (n_task, 1, n_classes)
        worker_labels_sum_expanded = worker_labels_sum[:, None, :]

        # off_diag_terms: (n_classes, n_classes) -> (1, n_classes, n_classes)
        off_diag_terms_expanded = off_diag_terms[None, :, :]

        powered_terms = off_diag_terms_expanded**worker_labels_sum_expanded

        off_diag_contrib_matrix = np.prod(powered_terms, axis=2)

        T = diag_contrib_matrix * off_diag_contrib_matrix

        self.denom_e_step = T.sum(axis=1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
