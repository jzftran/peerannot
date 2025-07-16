import numpy as np

from peerannot.models import DawidSkene


class PooledFlatSingleBinomial(DawidSkene):
    def _init_T(self) -> None:
        """
        Initialize posterior :math:`T_{i,\\l}` using normalized majority voting.

        Let:
        - :math:`n_{i,\\l}` = number of annotators labeling task :math:`i` as class :math:`\\l`
        - :math:`n_i = \\sum_{\\l=1}^K n_{i,\\l}` = total number of votes on task :math:`i`

        Then:

        .. math::
            T_{i,\\l} = \\frac{n_{i,\\l}}{n_i}
        """

        self.n_il = np.sum(
            self.crowd_matrix,
            axis=1,
        )  # n_tasks, n_classes: sum of votes given by each worker

        n_i = np.sum(
            self.n_il,
            axis=1,
            keepdims=True,
        )  # total number of votes per task

        self.total_votes = np.sum(self.n_il)

        self.T = np.divide(
            self.n_il,
            n_i,
            out=np.zeros_like(self.n_il, dtype=float),
            where=n_i != 0,
        )

    def _m_step(self) -> None:
        """
        M-step: maximize the expected complete log-likelihood under the shared accuracy model.

        Update estimates for:
        - Class priors:

        .. math::
            \\rho_{\\l} = \\frac{1}{n_{\\text{task}}} \\sum_{i=1}^{n_{\\text{task}}} T_{i,\\l}

        - Shared accuracy:

        .. math::
            \\alpha = \\frac{\\sum\\limits_{i=1}^{n_{\\text{task}}} \\sum\\limits_{\\l=1}^K T_{i,\\l} \\cdot n_{i,\\l}}{\\sum\\limits_{i=1}^{n_{\\text{task}}} n_i}
        """

        self.rho = self.T.sum(0) / self.n_task

        sum_diag_votes = np.einsum(
            "tq, tiq ->",
            self.T,
            self.crowd_matrix,
        )  # trace(T.T @ crowd_matrix)

        self.alpha = sum_diag_votes / self.total_votes

    def _e_step(self) -> None:
        T = np.zeros((self.n_task, self.n_classes))

        for i in range(self.n_task):
            n_i_k = self.n_il[i]  # shape (K,)

            for l in range(self.n_classes):
                p_lk = (1 - self.alpha) * self.rho
                p_lk[l] = self.alpha + (1 - self.alpha) * self.rho[l]

                T[i, l] = np.prod(np.power(p_lk, n_i_k))  # * self.rho[l]

        # Normalize
        self.denom_e_step = T.sum(axis=1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)


class PooledFlatSingleBinomialAlt(DawidSkene):
    def _init_T(self) -> None:
        """
        Initialize posterior :math:`T_{i,\\l}` using normalized majority voting.

        Let:
        - :math:`n_{i,\\l}` = number of annotators labeling task :math:`i` as class :math:`\\l`
        - :math:`n_i = \\sum_{\\l=1}^K n_{i,\\l}` = total number of votes on task :math:`i`

        Then:

        .. math::
            T_{i,\\l} = \\frac{n_{i,\\l}}{n_i}
        """

        self.n_il = np.sum(
            self.crowd_matrix,
            axis=1,
        )  # n_tasks, n_classes: sum of votes given by each worker

        n_i = np.sum(
            self.n_il,
            axis=1,
            keepdims=True,
        )  # total number of votes per task

        self.total_votes = np.sum(self.n_il)

        self.T = np.divide(
            self.n_il,
            n_i,
            out=np.zeros_like(self.n_il, dtype=float),
            where=n_i != 0,
        )

    def _m_step(self) -> None:
        """
        M-step: maximize the expected complete log-likelihood under the shared accuracy model.

        Update estimates for:
        - Class priors:

        .. math::
            \\rho_{\\l} = \\frac{1}{n_{\\text{task}}} \\sum_{i=1}^{n_{\\text{task}}} T_{i,\\l}

        - Shared accuracy:

        .. math::
            \\alpha = \\frac{\\sum\\limits_{i=1}^{n_{\\text{task}}} \\sum\\limits_{\\l=1}^K T_{i,\\l} \\cdot n_{i,\\l}}{\\sum\\limits_{i=1}^{n_{\\text{task}}} n_i}
        """

        self.rho = self.T.sum(0) / self.n_task

        sum_diag_votes = np.einsum(
            "tq, tiq ->",
            self.T,
            self.crowd_matrix,
        )  # trace(T.T @ crowd_matrix)

        self.alpha = sum_diag_votes / self.total_votes

    def _e_step(self) -> None:
        T = np.zeros((self.n_task, self.n_classes))

        for i in range(self.n_task):
            n_i_k = self.n_il[i]  # shape (K,)

            for l in range(self.n_classes):
                p_lk = ((1 - self.alpha) * self.rho) / (1 - self.rho[l])
                p_lk[l] = self.alpha

                T[i, l] = np.prod(np.power(p_lk, n_i_k))  # * self.rho[l]

        # Normalize
        self.denom_e_step = T.sum(axis=1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
