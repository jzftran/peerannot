import numpy as np

from .dawid_skene import DawidSkene


class PooledDiagonalMultinomial(DawidSkene):
    """
    ===========================
    Pooled Diagonal Multinomial
    ===========================

    A simplified and pooled variant of the Dawid and Skene model (1979),
    assuming **all workers share the same confusion matrix** structure,
    with only **diagonal values estimated individually**, and
    off-diagonal elements being uniform and shared across all classes.

    This model assumes that workers are interchangeable, and only the
    aggregate reliability for each class is learned.

    Assumptions:

    - Workers are the same. The model is class-based, not worker-based: confusion structure is shared.

    - Each class has a single scalar "correct labeling probability" (:math:`\\pi_k`).

    - All off-diagonal probabilities are pooled: uniformly distributed error.

    - Off-diagonal entries are computed as:

      .. math::
          \\pi^{\\text{off}}_k = \\frac{1 - \\pi_k}{K - 1}

    Using:

    - EM algorithm


    Estimating:

    - One diagonal of the confusion matrix, shared between workers.

    
    For a worker labeling task :math:`i` as class :math:`\\ell`, assuming
    the true class is :math:`k`, the model assumes:

    .. math::

        P(y_i^{(j)} = \\ell \\mid y_i^* = k) =
        \\begin{cases}
            \\pi_k & \\text{if } \\ell = k \\\\
            \\frac{1 - \\pi_k}{K - 1} & \\text{otherwise}
        \\end{cases}


    where:
    
    - :math:`y_i^{\\star} \\in \\{0, ..., K-1\\}` is the true class label for task :math:`i`
    
    - :math:`\\pi_k \\in [0,1]` is the probability of correctly labeling class :math:`k`
    
    - :math:`K` is the number of classes

    """

    def _m_step(self) -> None:
        """
        M-step of the EM algorithm: update class priors :math:`\\rho_k` and pooled diagonal accuracies :math:`\\pi_k`.

        **Class priors**:

        .. math::
            \\rho_k = \\frac{1}{n} \\sum_{i=1}^{n} T_{i,k}

        **Pooled diagonal accuracies**:

        For each class :math:`k`, we compute a single shared accuracy :math:`\\pi_k`
        over all workers jointly (pooled):

        .. math::
            \\pi_k = \\frac{\\sum_{i=1}^{n} T_{i,k} \\cdot \\sum_{j=1}^{W} \\mathbb{1}(y_i^{(j)} = k)}
                        {\\sum_{i=1}^{n} T_{i,k} \\cdot \\sum_{j=1}^{W} \\sum_{\\ell=1}^{K} \\mathbb{1}(y_i^{(j)} = \\ell)}

        **Off-diagonal probability** (uniformly distributed across the other classes):

        .. math::
            \\pi^{\\text{off}}_k = \\frac{1 - \\pi_k}{K - 1}

        """

        self.rho = self.T.sum(0) / self.n_task

        diag_votes = np.einsum("tq, tiq -> q", self.T, self.crowd_matrix)
        denom = np.einsum("tq, tij -> q", self.T, self.crowd_matrix)

        self.pi = np.divide(
            diag_votes,
            denom,
            out=np.zeros_like(diag_votes),
            where=denom != 0,
        )

        self.pi_non_diag_values = (np.ones_like(self.pi) - self.pi) / (
            self.n_classes - 1
        )

    def _e_step(self) -> None:
        """
        E-step of the EM algorithm: estimate posterior probabilities :math:`T_{i,k}` for all tasks and classes.

        For each task :math:`i` and class :math:`k`, we estimate:

        .. math::
            T_{i,k} = \\frac{
                \\rho_k \\cdot
                \\prod_{j=1}^W
                \\left[
                    \\pi_k^{\\mathbb{1}(y_i^{(j)} = k)} \\cdot
                    \\left(\\frac{1 - \\pi_k}{K - 1}\\right)^{\\sum_{\\ell \\neq k} \\mathbb{1}(y_i^{(j)} = \\ell)}
                \\right]
            }{
                \\sum_{k'=1}^K \\rho_{k'} \\cdot
                \\prod_{j=1}^W
                \\left[
                    \\pi_{k'}^{\\mathbb{1}(y_i^{(j)} = k')} \\cdot
                    \\left(\\frac{1 - \\pi_{k'}}{K - 1}\\right)^{\\sum_{\\ell \\neq k'} \\mathbb{1}(y_i^{(j)} = \\ell)}
                \\right]
            }

        Where:
            - :math:`\\pi_k` is the estimated accuracy when the true class is :math:`k` (shared across all workers),
            - :math:`y_i^{(j)}` is the label provided by worker :math:`j` for task :math:`i`,
            - :math:`K` is the number of classes and :math:`W` is the number of workers,
            - :math:`\\rho_k` is the prior probability of class :math:`k`.


        """

        T = np.zeros((self.n_task, self.n_classes))

        for i in range(self.n_task):
            for j in range(self.n_classes):
                worker_labels = self.crowd_matrix[i]
                diag_contrib = np.prod(np.power(self.pi, worker_labels))
                mask = np.ones(self.n_classes, dtype=bool)
                mask[j] = False
                off_diag_contrib = np.prod(
                    np.power(
                        self.pi_non_diag_values[mask],
                        worker_labels[:, mask],
                    ),
                )

                T[i, j] = diag_contrib * off_diag_contrib * self.rho[j]

        self.denom_e_step = T.sum(1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)
