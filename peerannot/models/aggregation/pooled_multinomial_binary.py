import numpy as np

from peerannot.models import DawidSkene


class PoooledMultinomialBinary(DawidSkene):
    """
    =========================
    Pooled Multinomial Binary
    =========================


    A simplified variant of the Dawid & Skene model where:
        - A single accuracy parameter (`alpha`) is used across all workers.
        - Off-diagonal errors are assumed to be uniformly distributed.
        - Votes are pooled across all workers (no per-worker confusion matrix).

    Assumptions:
        - Workers are the same.
        - Diagonal entries (correct labels) use a single shared alpha value.
        - Errors are symmetric across all incorrect labels (off-diagonal entries are equal).
        - Labels are conditionally independent given the true label.


    Suitable for low-data regimes or settings with homogeneous labelers.
    """

    def _init_T(self) -> None:
        """
        Initialize posterior :math:`T_{i,\\ell}` using normalized majority voting.

        Let:
        - :math:`n_{i,\\ell}` = number of annotators labeling task :math:`i` as class :math:`\\ell`
        - :math:`n_i = \\sum_{\\ell=1}^K n_{i,\\ell}` = total number of votes on task :math:`i`

        Then:

        .. math::
            T_{i,\\ell} = \\frac{n_{i,\\ell}}{n_i}
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
            \\rho_{\\ell} = \\frac{1}{n_{\\text{task}}} \\sum_{i=1}^{n_{\\text{task}}} T_{i,\\ell}

        - Shared accuracy:

        .. math::
            \\alpha = \\frac{\\sum\\limits_{i=1}^{n_{\\text{task}}} \\sum\\limits_{\\ell=1}^K T_{i,\\ell} \\cdot n_{i,\\ell}}{\\sum\\limits_{i=1}^{n_{\\text{task}}} n_i}
        """

        self.rho = self.T.sum(0) / self.n_task

        sum_diag_votes = np.einsum(
            "tq, tiq ->",
            self.T,
            self.crowd_matrix,
        )  # trace(T.T @ crowd_matrix)

        self.alpha = sum_diag_votes / self.total_votes

    def _e_step(self) -> None:
        """
        E-step: update soft-labels :math:`T_{i,\\ell}` using current :math:`\\rho` and :math:`\\alpha`.

        For each task :math:`i` and label :math:`\\ell`, compute the unnormalized posterior:

        Let:
        - :math:`n_{i,\\ell}` = number of annotators who labeled task :math:`i` as class :math:`\\ell`
        - :math:`n_i` = total annotations on task :math:`i`

        Then:

        .. math::
            T_{i,\\ell} =
            \\frac{
                \\rho_{\\ell} \\cdot
                \\alpha^{n_{i,\\ell}} \\cdot
                \\left( \\frac{1 - \\alpha}{K - 1} \\right)^{n_i - n_{i,\\ell}}
            }{
                \\sum\\limits_{\\ell'=1}^K
                \\rho_{\\ell'} \\cdot
                \\alpha^{n_{i,\\ell'}} \\cdot
                \\left( \\frac{1 - \\alpha}{K - 1} \\right)^{n_i - n_{i,\\ell'}}
            }
        """

        T = np.zeros((self.n_task, self.n_classes))

        for i in range(self.n_task):
            n_i = self.n_il[i].sum()  # total numer of annotators of task i
            for l in range(self.n_classes):
                n_il = self.n_il[
                    i,
                    l,
                ]  # numer of annotators of task i voting for label l
                diag_contrib = np.power(self.alpha, n_il)
                off_diag_contrib = np.power(
                    (1 - self.alpha) / (self.n_classes - 1),
                    n_i - n_il,
                )

                T[i, l] = diag_contrib * off_diag_contrib * self.rho[l]

        self.denom_e_step = T.sum(axis=1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)


class VectorizedPoooledMultinomialBinary(PoooledMultinomialBinary):
    def _e_step(self):
        n_i = self.n_il.sum(axis=1, keepdims=True)

        diag_contrib = self.alpha**self.n_il

        off_diag_factor = (1 - self.alpha) / (self.n_classes - 1)
        off_diag_contrib = off_diag_factor ** (n_i - self.n_il)

        T = diag_contrib * off_diag_contrib * self.rho[np.newaxis, :]

        self.denom_e_step = T.sum(axis=1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
