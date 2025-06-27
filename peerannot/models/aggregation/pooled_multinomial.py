import numpy as np

from .dawid_skene import DawidSkene


class PooledMultinomial(DawidSkene):
    """
    ==================
    Pooled Multinomial
    ==================

    Variation of the Dawid and Skene model (1979)
    with a single, shared confusion matrix between all users.

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
        """
        M-step of the EM algorithm: estimate class priors
        :math:`\\rho_k` and a shared confusion matrix :math:`\\pi_{k, \\ell}`.

        **Class priors**:

        .. math::
            \\rho_k = \\frac{1}{n} \\sum_{i=1}^{n} T_{i,k}

        **Shared confusion matrix**:

        All workers are treated as having a common confusion matrix :math:`\\pi_{k, \\ell}`,
        where:

        - :math:`k` is the true class,
        - :math:`\\ell` is the observed label,
        - :math:`\\pi_{k, \\ell}` = probability of a worker labeling a task as :math:`\\ell` when its true label is :math:`k`.

        Estimates:

        .. math::
            \\pi_{k, \\ell} = \\frac{\\sum_{i=1}^{n} T_{i,k} \\cdot \\sum_{j=1}^{W} \\mathbb{1}(y_i^{(j)} = \\ell)}
                            {\\sum_{i=1}^{n} T_{i,k} \\cdot \\sum_{j=1}^{W} 1}

        """

        self.rho = self.T.sum(0) / self.n_task

        aggregated_votes = np.einsum(
            "tq, tij -> qj",
            self.T,
            self.crowd_matrix,
        )  # shape (n_classes, n_classes)
        denom = aggregated_votes.sum(axis=1, keepdims=True)

        self.shared_pi = np.divide(
            aggregated_votes,
            denom,
            out=np.zeros_like(aggregated_votes),
            where=denom != 0,
        )

    def _e_step(self) -> None:
        """
        E-step of the EM algorithm: estimate posterior probabilities :math:`T_{i,k}` of task :math:`i` having true label :math:`k`.

        For each task :math:`i` and each class :math:`k`, compute:


        .. math::
            T_{i,k} = \\frac{
                \\rho_k \\cdot
                \\prod_{j=1}^{W}
                \\prod_{\\ell=1}^{K}
                \\pi_{k, \\ell}^{\\mathbb{1}(y_i^{(j)} = \\ell)}
            }{
                \\sum_{k'=1}^{K}
                \\rho_{k'} \\cdot
                \\prod_{j=1}^{W}
                \\prod_{\\ell=1}^{K}
                \\pi_{k', \\ell}^{\\mathbb{1}(y_i^{(j)} = \\ell)}
            }

        where:
            - :math:`\\pi_{k, \\ell}` is the probability of a worker labeling a task with true class :math:`k` as class :math:`\\ell`
              (i.e. entry in the shared confusion matrix),
            - :math:`y_i^{(j)}` is the label provided by worker :math:`j` on task :math:`i`,
            - :math:`\\rho_k` is the prior probability of class :math:`k`,
            - :math:`W` is the number of workers, and :math:`K` is the number of classes.

        """

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
