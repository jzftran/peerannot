import warnings
from os import PathLike
from sys import getsizeof
from typing import Annotated, Generator

import numpy as np
import sparse as sp
from annotated_types import Ge
from loguru import logger
from pydantic import validate_call
from tqdm.auto import tqdm

from peerannot.models.aggregation.warnings import DidNotConverge
from peerannot.models.template import AnswersDict, CrowdModel

FilePathInput = PathLike | str | list[str] | Generator[str, None, None] | None
SPARSE_NOT_IMPLEMENTED = "The sparse algorithm for DawidSkeneShared\
                in the current context is not implemented."


class PooledMultinomial(CrowdModel):
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

    @validate_call
    def __init__(
        self,
        answers: AnswersDict,
        # TODO@jzftran probably annotation with 0 or 1 worker doesn't make sense: for 0 it should be an error
        n_workers: Annotated[int, Ge(1)],
        n_classes: Annotated[int, Ge(1)],
        *,
        sparse: bool = False,
        path_remove: FilePathInput = None,
    ) -> None:
        r"""Dawid and Skene strategy: estimate confusion matrix for each worker.

        Assuming that workers are independent, the model assumes that

        .. math::

            (y_i^{(j)}\ | y_i^\\star = k) \\sim \\mathcal{M}\\left(\\pi^{(j)}_{k,\\cdot}\\right)

        and maximizes the log likelihood of the model using an EM algorithm.

        .. math::

            \\underset{\\rho,\\\pi,T}{\mathrm{argmax}}\\prod_{i\\in [n_{\\texttt{task}}]}\prod_{k \\in [K]}\\bigg[\\rho_k\prod_{j\\in [n_{\\texttt{worker}}]}\prod_{\\ell\in [K]}\\big(\\pi^{(j)}_{k, \\ell}\\big)^{\mathbf{1}_{\\{y_i^{(j)}=\\ell\\}}}\\bigg]^{T_{i,k}},

        where :math:`\\rho` is the class marginals, :math:`\\pi` is the confusion matrix and :math:`T` is the indicator variables of belonging to each class.

        :param answers: Dictionary of workers answers with format

         .. code-block:: javascript

            {
                task0: {worker0: label, worker1: label},
                task1: {worker1: label}
            }

        :type answers: dict
        :param sparse: If the number of workers/tasks/label is large (:math:`>10^{6}` for at least one), use sparse=True to run per task
        :type sparse: bool, optional
        :param n_classes: Number of possible classes, defaults to 2
        :type n_classes: int, optional"""

        super().__init__(answers)
        self.n_workers: int = n_workers
        self.n_classes: int = n_classes
        self.sparse: bool = sparse
        self.path_remove: FilePathInput = path_remove
        self.n_task: int = len(self.answers)

        self.exclude_answers()
        if self.sparse:
            raise NotImplementedError(SPARSE_NOT_IMPLEMENTED)
        else:
            self.init_crowd_matrix()

    def exclude_answers(self) -> None:
        answers_modif = {}
        if self.path_remove is not None:
            to_remove = np.loadtxt(self.path_remove, dtype=int)
            i = 0
            for key, val in self.answers.items():
                if int(key) not in to_remove[:, 1]:
                    answers_modif[i] = val
                    i += 1
            self.answers = answers_modif

    def init_crowd_matrix(self) -> None:
        """Transform dictionnary of labels to a tensor of size
        (n_task, n_workers, n_classes)."""

        matrix = np.zeros((self.n_task, self.n_workers, self.n_classes))
        for task, ans in self.answers.items():
            for worker, label in ans.items():
                matrix[task, worker, label] += 1

        logger.debug(f"Dense crowd matrix  {getsizeof(matrix)}")
        self.crowd_matrix = matrix

    def init_T(self):
        """NS initialization"""
        T = self.crowd_matrix.sum(axis=1)  # shape (n_task, n_classes)

        logger.debug(f"Size of T before calc: {getsizeof(T)}")

        tdim = T.sum(1, keepdims=True)
        self.T = np.where(tdim > 0, T / tdim, 0)
        logger.debug(f"Size of T: {getsizeof(self.T)}")

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
            "tq, tij -> qj", self.T, self.crowd_matrix
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
                            self.shared_pi[j, :], self.crowd_matrix[i, :, :]
                        )
                    )
                    * self.rho[j]
                )
                T[i, j] = num

        self.denom_e_step = T.sum(axis=1, keepdims=True)
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)

    def log_likelihood(self):
        """Compute log likelihood of the model"""
        return np.log(np.sum(self.denom_e_step))

    def run_dense(
        self,
        epsilon: Annotated[float, Ge(0)] = 1e-6,
        maxiter: Annotated[int, Ge(0)] = 50,
        *,
        verbose: bool = False,
    ) -> tuple[list[np.float64], int]:
        i = 0
        eps = np.inf

        self.init_T()
        ll = []
        pbar = tqdm(total=maxiter, desc="Dawid and Skene")
        while i < maxiter and eps > epsilon:
            self._m_step()
            self._e_step()
            likeli = self.log_likelihood()
            ll.append(likeli)
            if len(ll) >= 2:
                eps = np.abs(ll[-1] - ll[-2])
            i += 1
            pbar.update(1)

        pbar.set_description("Finished")
        pbar.close()
        self.c = i
        if eps > epsilon:
            warnings.warn(str(eps), DidNotConverge, stacklevel=2)
        return ll, i

    @validate_call
    def run(
        self,
        epsilon: Annotated[float, Ge(0)] = 1e-6,
        maxiter: Annotated[int, Ge(0)] = 50,
        *,
        verbose: bool = False,
    ) -> tuple[list[np.float64], int]:
        """Run the EM optimization

        :param epsilon: stopping criterion (:math:`\\ell_1` norm between two iterates of log likelihood), defaults to 1e-6
        :type epsilon: float, optional
        :param maxiter: Maximum number of steps, defaults to 50
        :type maxiter: int, optional
        :param verbose: Verbosity level, defaults to False
        :type verbose: bool, optional
        :return: Log likelihood values and number of steps taken
        :rtype: (list,int)
        """

        if self.sparse:
            raise NotImplementedError(SPARSE_NOT_IMPLEMENTED)
        return self.run_dense(
            epsilon=epsilon,
            maxiter=maxiter,
            verbose=verbose,
        )

    def get_answers(self):
        """Get most probable labels"""
        if self.sparse:
            return np.vectorize(self.inv_labels.get)(
                sp.argmax(self.sparse_T, axis=1).todense()
            )
        return np.vectorize(self.inv_labels.get)(
            np.argmax(self.get_probas(), axis=1)
        )

    def get_probas(self):
        """Get soft labels distribution for each task"""
        return self.T
