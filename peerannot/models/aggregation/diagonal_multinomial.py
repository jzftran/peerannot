import numpy as np

from .dawid_skene import DawidSkene


class DiagonalMultinomial(DawidSkene):
    """
    ====================
    Diagonal Multinomial
    ====================

    A simplified variant of the Dawid and Skene model (1979),
    assuming workers only make diagonal errors in the confusion matrix.


    Assumptions:

    - workers are independent

    - each worker is only characterized by their reliability in recognizing the correct class

    - all errors (misclassifications) are uniformly distributed among the incorrect classes

    Using:

    - EM algorithm

    Estimating:

    - One diagonal of the confusion matrix for each worker



    For a worker $j$ labeling a task $i$, we assume:

    .. math::

        P(y_i^{(j)} = \\ell \\mid y_i^* = k) = \\begin{cases} \\pi_{j, k} & \\text{if} \\ell = k \\\\
        \\frac{1 - \\pi_{j, k}}{K - 1} & \\text{otherwise} \\end{cases}



    where:

    * $y_i^{\\star} \\in \\{0, ..., K-1\\}$ is the true class label for task $i$,
    * $\\pi_{j,k} \\in [0,1]$ is the probability that worker $j$ correctly identifies class $k$,
    * $K$ is the number of classes.

    This model keeps only the diagonal of each worker's confusion matrix and assumes the rest is constant.




    """

    def _m_step(self) -> None:
        """
        Update parameters :math:`\\rho_k` and :math:`\\pi_{j,k}` using current posterior :math:`T`.

        **Class priors**:

        :math:`\\rho_k = \\frac{1}{n} \\sum_{i=1}^{n} T_{i,k}`

        **Worker diagonal accuracies**:

        .. math::

            \\pi_{j,k} = \\frac{\\sum_{i=1}^{n} T_{i,k} \\cdot \\mathbb{1}_{y_i^{(j)} = k}}{\\sum_{i=1}^{n} T_{i,k} \\cdot \\sum_{\\ell} \\mathbb{1}_{y_i^{(j)} = \\ell}}


        This gives the fraction of times worker :math:`j` labels class :math:`k` correctly when it is estimated to be class :math:`k`.

        * **Off-diagonal errors**:

        .. math::

            \\pi^{\\text{off}}_{j,k} = \\frac{1 - \\pi_{j,k}}{K - 1}

        """
        rho = self.T.sum(0) / self.n_task

        pi = np.zeros((self.n_workers, self.n_classes))
        for j in range(self.n_classes):
            # TODO @jzftran: change this to calculate diagonal faster
            pij = self.T[:, j] @ self.crowd_matrix.transpose((1, 0, 2))
            denom = pij.sum(1)

            diag_values = pij[:, j] / np.where(denom > 0, denom, 1e-9)
            pi[:, j] = diag_values

        # pi shape (n_workers, n_class), reresents how sure worker is
        # sure that the label j is true
        pi_non_diag_values = (np.ones_like(pi) - pi) / (self.n_classes - 1)
        self.rho, self.pi, self.pi_non_diag_values = (
            rho,
            pi,
            pi_non_diag_values,
        )

    def _e_step(self) -> None:
        """

        Estimate the posterior probability :math:`T_{i,k}` that task :math:`i`
        belongs to class :math:`k`, given current estimates of class marginals
        :math:`\\rho_k` and diagonal worker accuracies :math:`\\pi_{j,k}`:

        .. math::

            T_{i,k} \\propto \\rho_k \\prod_{j=1}^{W}\\left[ \\pi_{j,k}^{\\mathbb{1}(y_i^{(j)} = k)} \\cdot \\left( \\frac{1 - \\pi_{j,k}}{K - 1} \\right)^{\\sum_{\\ell \\neq k} \\mathbb{1}(y_i^{(j)} = \\ell)} \\right]

        Normalize over :math:`k \\in \\{0, ..., K-1\\}` to get :math:`T_{i,k}`.

        """

        T = np.zeros((self.n_task, self.n_classes))

        for i in range(self.n_task):
            for j in range(self.n_classes):
                # Get all worker labels for task i (n_workers, n_classes)
                worker_labels = self.crowd_matrix[
                    i
                ]  # shape (n_workers, n_classes)

                # Vectorized computation for all workers simultaneously
                # Diagonal contributions: pi[k,j]^worker_labels[k,j]
                diag_contrib = np.power(
                    self.pi[:, j],
                    worker_labels[:, j],
                )  # shape (n_workers,)

                # Off-diagonal contributions: product over l≠j of
                # pi_non_diag[k,j]^worker_labels[k,l]
                mask = np.ones(self.n_classes, dtype=bool)
                mask[j] = False  # exclude current class j
                off_diag_labels = worker_labels[
                    :,
                    mask,
                ]  # shape (n_workers, n_classes-1)

                off_diag_contrib = np.prod(
                    np.power(
                        self.pi_non_diag_values[:, j][:, np.newaxis],
                        off_diag_labels,
                    ),
                    axis=1,
                )  # shape (n_workers,)

                worker_probs = (
                    diag_contrib * off_diag_contrib
                )  # shape (n_workers,)
                T[i, j] = np.prod(worker_probs) * self.rho[j]

        self.denom_e_step = T.sum(1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)


class VectorizedDiagonalMultinomial(DiagonalMultinomial):
    """
    A vectorized and optimized version of `DiagonalMultinomial`,
    implementing the same model assumptions, but using
    NumPy broadcasting (`einsum`, masking, etc.) for speed.
    """

    def _m_step(self) -> None:
        rho = self.T.sum(0) / self.n_task
        pij = np.einsum("tq,tiq->iq", self.T, self.crowd_matrix)
        denom = np.einsum("tq, tij -> iq", self.T, self.crowd_matrix)
        pi = np.where(denom > 0, pij / denom, 0)
        # pi shape (n_workers, n_classes)
        pi_non_diag_values = (np.ones_like(pi) - pi) / (self.n_classes - 1)

        self.rho, self.pi, self.pi_non_diag_values = (
            rho,
            pi,
            pi_non_diag_values,
        )

    def _e_step(self) -> None:
        # Compute diagonal contributions
        # shape: (n_task, n_workers, n_classes)
        diag_contrib = np.power(
            self.pi[np.newaxis, :, :],  # (1, n_workers, n_classes)
            self.crowd_matrix,  # (n_task, n_workers, n_classes)
        )

        # Compute off-diagonal contributions
        # For each class j, we need to multiply pi_non_diag_values[k,j]^worker_labels[l] for all l != j
        mask = 1 - np.eye(self.n_classes)  # (n_casses, n_classes)

        # shape: (n_task, n_workers, n_classes, n_classes)
        off_diag_powers = np.power(
            self.pi_non_diag_values[
                np.newaxis,
                :,
                np.newaxis,
                :,
            ],  # (1, n_workers, 1, n_classes)
            self.crowd_matrix[:, :, :, np.newaxis]
            * mask[np.newaxis, np.newaxis, :, :],
        )

        off_diag_contrib = np.prod(
            off_diag_powers,
            axis=2,
        )  # (n_task, n_workers, n_classes)

        worker_probs = diag_contrib * off_diag_contrib

        T = (
            np.prod(worker_probs, axis=1) * self.rho[np.newaxis, :]
        )  # (n_task, n_classes)

        self.denom_e_step = T.sum(axis=1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
