from __future__ import annotations

import warnings
from typing import (
    Annotated,
)

import numpy as np
from annotated_types import Ge, Gt
from line_profiler import profile

from peerannot.models.aggregation.mongo_online_helpers import (
    SparseMongoBatchAlgorithm,
)
from peerannot.models.aggregation.warnings_errors import (
    DidNotConverge,
)


class MajorityVotingMongo(SparseMongoBatchAlgorithm):
    """
    =========================
    Majority voting
    =========================
    Most answered label per task
    """

    @profile
    def _em_loop_on_batch(
        self,
        batch_matrix: np.ndarray,
        epsilon: Annotated[float, Ge(0)] = 1e-6,
        maxiter: Annotated[int, Gt(0)] = 50,
    ) -> list[float]:
        i = 0
        eps = np.inf
        ll: list[float] = []

        batch_T = self._init_T(
            batch_matrix,
        )

        if eps > epsilon:
            warnings.warn(
                DidNotConverge(self.__class__.__name__, eps, epsilon),
                stacklevel=2,
            )

        # Online updates
        batch_rho = np.array([])
        batch_pi = np.array([])
        batch_rho = np.zeros(len(self._batch_class_to_idx))
        self._online_update(
            batch_T,
            batch_rho,
            batch_pi,
        )
        ll = [0, 0]
        return ll

    def _online_update_pi(self, batch_pi):
        pass

    def _e_step(self, batch_matrix, batch_pi, batch_rho):
        pass

    def _m_step(self, batch_matrix, batch_T):
        pass

    def pi(self):
        pass
