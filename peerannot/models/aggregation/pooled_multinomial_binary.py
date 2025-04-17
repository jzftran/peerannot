import numpy as np

from peerannot.models import DawidSkene


class PoooledMultinomialBinary(DawidSkene):
    def _init_T(self) -> None:
        # T shape n_tasks, n classes
        self.n_il = np.sum(
            self.crowd_matrix,
            axis=1,
        )  # n_tasks, n_classes: sum of votes given by each worker

        n_i = np.sum(self.n_il, axis=0)

        self.total_votes = np.sum(self.n_il)

        self.T = self.n_il / n_i

    def _m_step(self) -> None:
        """Maximizing log likelihood with a single confusion matrix shared
        across all workers."""

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
