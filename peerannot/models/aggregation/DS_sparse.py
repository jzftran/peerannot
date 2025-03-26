from .DS import DawidSkene
import warnings
from os import PathLike
from sys import getsizeof
from typing import Annotated, Generator

import numpy as np
import sparse as sp
from annotated_types import Ge
from loguru import logger
from numpy.typing import NDArray
from pydantic import validate_call
from tqdm.auto import tqdm

from peerannot.models.aggregation.warnings import DidNotConverge
from peerannot.models.template import AnswersDict, CrowdModel


class DawidSkeneSparse(DawidSkene):
    def __init__(self, answers, n_workers, n_classes, *, path_remove=None):
        super().__init__(
            answers, n_workers, n_classes, path_remove=path_remove
        )

    def init_crowd_matrix(self) -> None:
        """Transform dictionnary of labels to a tensor of size
        (n_task, n_workers, n_classes)."""
        # TODO crowd matrix usually will be sparse, maybe there is another
        #  better implementation for it
        crowd_matrix = sp.DOK(
            (self.n_task, self.n_workers, self.n_classes), dtype=np.uint16
        )

        for task, ans in self.answers.items():
            for worker, label in ans.items():
                crowd_matrix[task, worker, label] = 1

        logger.debug(f"Sparse crowd matrix {getsizeof(crowd_matrix)}")
        self.crowd_matrix = crowd_matrix.to_coo()

    def init_T(self) -> None:
        """NS initialization"""
        # T shape is n_task, n_classes
        T = self.crowd_matrix.sum(axis=1)
        logger.debug(f"Size of T before calc: {getsizeof(T)}")

        tdim = T.sum(1, keepdims=True).todense()
        self.T = np.where(tdim > 0, T / tdim, 0)
        logger.debug(f"Size of T: {getsizeof(self.T)}")

    def _m_step_sparse(
        self,
    ) -> Generator[NDArray, None, None]:
        """Maximizing log likelihood (see eq. 2.3 and 2.4 Dawid and Skene 1979)

        Returns:
            :math:`\\rho`: :math:`(\\rho_j)_j` probabilities that instance has true response j if drawn at random (class marginals)
            pi: number of times worker k records l when j is correct
        """
        # pi could be bigger, at least inner 2d matrices should be implemented as sparse, probably the easiest way to create is to use dok array

        self.rho = self.T.sum(axis=0) / self.n_task  # can rho be sparse?

        transposed_sparse_crowd_matrix = self.crowd_matrix.transpose(
            (1, 0, 2),
        )
        # Compute sparse confusion matrices
        for q in range(self.n_classes):
            pij = self.T[:, q] @ transposed_sparse_crowd_matrix
            denom = pij.tocsr().sum(1)
            safe_denom = np.where(denom <= 0, -1e9, denom).reshape(-1, 1)
            yield pij / safe_denom

    def _e_step(self) -> None:
        """Estimate indicator variables (see eq. 2.5 Dawid and Skene 1979)

        Returns:
            T: New estimate for indicator variables (n_task, n_worker)
            denom: value used to compute likelihood easily
        """
        T = sp.DOK(shape=(self.n_task, self.n_classes))

        m_step = self._m_step_sparse()

        for j, pij in enumerate(m_step):
            for i in range(self.n_task):
                num = (
                    np.prod(np.power(pij, self.crowd_matrix[i])) * self.rho[j]
                )
                T[i, j] = num

        T = T.to_coo()
        self.denom_e_step = T.sum(1, keepdims=True).todense()
        self.T = np.where(self.denom_e_step > 0, T / self.denom_e_step, T)

    @validate_call
    def run(
        self,
        epsilon: Annotated[float, Ge(0)] = 1e-6,
        maxiter: Annotated[int, Ge(0)] = 50,
        *,
        verbose: bool = False,
    ) -> tuple[list[float], int]:
        i = 0
        eps = np.inf

        self.init_T()
        ll = []
        pbar = tqdm(total=maxiter, desc="Dawid and Skene Sparse")
        while i < maxiter and eps > epsilon:
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

    def get_answers(self) -> NDArray:
        """Get most probable labels"""

        return np.vectorize(self.inv_labels.get)(
            sp.argmax(self.T, axis=1).todense(),
        )
