import numpy as np

from peerannot.models import DawidSkene


class MultinomialBinary(DawidSkene):
    """
    ===================
    Multinomial Binary
    ===================

    A simplified variation of the Dawid & Skene (1979) model where:

    - Each worker's labeling behavior is characterized by a single accuracy value
        (diagonal value alpha),
    - All incorrect labels (off-diagonal) are uniformly distributed across
        the remaining classes,
    - Off-diagonal error rates are derived from alpha and are shared across all classes,
    - Workers are assumed to be independent.


    Assumptions:
        - independent workers
        - worker can be either good or bad
    Using:
        - EM algorithm

    Esimating:
        - One single value for each worker

    """

    def _m_step(self) -> None:
        """
        M-step of the EM algorithm.

        Maximizes the expected log-likelihood with respect to:
        - Class prior probabilities :math:`\\rho`
        - Per-worker accuracy parameters :math:`\\pi_j` (diagonal values only)

        For each worker :math:`j`, we estimate accuracy as:

        .. math::
            \\pi_j = \\frac{
                \\sum_{i,k} T_{i,k} \\cdot \\mathbb{1}(y_i^{(j)} = k)
            }{
                \\sum_{i} \\sum_{k} \\mathbb{1}(y_i^{(j)} = k)
            }

        The off-diagonal confusion values are set uniformly as:

        .. math::
            \\pi_{j, \\ell \\neq k} = \\frac{1 - \\pi_j}{K - 1}

        Updates:
            - `self.rho` : Class priors (shape: n_classes)
            - `self.pi` : Per-worker diagonal accuracy (shape: n_workers)
            - `self.off_diag_alpha` : Uniform off-diagonal error values (shape: n_workers)
        """
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
        """
        E-step of the EM algorithm.

        Estimates posterior probabilities :math:`T_{i,k}` for each task :math:`i`
        and class :math:`k`, using current estimates of `rho`, `pi`, and
        `off_diag_alpha`.

        For each task and class, we compute:

        .. math::
            T_{i,k} =
                \\frac{
                    \\rho_k \\cdot \\prod_{j=1}^{W}
                    \\left[
                        \\pi_j^{\\mathbb{1}(y_i^{(j)} = k)} \\cdot
                        \\left( \\frac{1 - \\pi_j}{K - 1} \\right)^{\\sum_{\\ell \\neq k} \\mathbb{1}(y_i^{(j)} = \\ell)}
                    \\right]
                }{
                    \\sum_{k'=1}^{K}
                    \\rho_{k'} \\cdot \\prod_{j=1}^{W}
                    \\left[
                        \\pi_j^{\\mathbb{1}(y_i^{(j)} = k')} \\cdot
                        \\left( \\frac{1 - \\pi_j}{K - 1} \\right)^{\\sum_{\\ell \\neq k'} \\mathbb{1}(y_i^{(j)} = \\ell)}
                    \\right]
                }

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
