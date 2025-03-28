import numpy as np

from .DS import DawidSkene


class PooledMultinomial(DawidSkene):
    """
    ===============================================
    Variation of the Dawid and Skene model (1979)
    with a single, shared confusion matrix.
    ===============================================

    Assumptions:
    - all workers are treated as equally reliable

    Using:
    - EM algorithm

    Estimating:
    - One shared confusion matrix for workers
    """

    def _m_step(
        self,
    ) -> None:
        """Maximizing log likelihood (see eq. 2.3 and 2.4 Dawid and Skene 1979)

        Returns:
            :math:`\\rho`: :math:`(\\rho_j)_j` probabilities that instance has true response j if drawn at random (class marginals)
            pi: number of times worker k records l when j is correct
        """
        # TODO:@jzftran adapt docstring
        self.rho = self.T.sum(0) / self.n_task

        aggregated_votes = np.einsum(
            "tq, tij -> qj",
            self.T,
            self.crowd_matrix,
        )  # shape (n_classes, n_classes)
        denom = aggregated_votes.sum(axis=1, keepdims=True)
        self.shared_pi = np.where(denom > 0, aggregated_votes / denom, 0)

    def _e_step(self) -> None:
        """Estimate indicator variables using a shared confusion matrix"""

        T = np.zeros((self.n_task, self.n_classes))

        # use mask instead of power
        for i in range(self.n_task):
            for j in range(self.n_classes):
                num = (
                    np.prod(
                        np.power(
                            self.shared_pi[j, :],
                            self.crowd_matrix[i, :, :],
                        ),
                    )
                    * self.rho[j]
                )
                T[i, j] = num

        self.denom_e_step = T.sum(axis=1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
