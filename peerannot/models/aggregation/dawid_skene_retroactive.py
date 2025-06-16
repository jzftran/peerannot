from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Hashable

    from peerannot.models.template import AnswersDict
from functools import wraps
from typing import (
    TYPE_CHECKING,
)

from peerannot.models.aggregation.dawid_skene_online import (
    DawidSkeneOnline,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from peerannot.models.template import AnswersDict


def limit_recursion(max_depth: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            depth = kwargs.pop("_depth", 0)
            if depth > max_depth:
                msg = f"Max recursion depth {max_depth} exceeded"
                raise RecursionError(msg)
            return func(*args, **kwargs, _depth=depth + 1)

        return wrapper

    return decorator


class DawidSkeneRetroactive(DawidSkeneOnline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Store past observations as a list of tuples (task_id, worker_id, class_id)
        # Should this be stored as crowd_matrix?
        # TODO:@jzftran Explore options for storing this data in a file while
        # maintaining a reference (pointer) to it.
        self.past_observations: list[tuple[Hashable, Hashable, Hashable]] = []

        # Store the previous estimates of task true class distributions to detect changes
        # stores previous T
        self.prev_task_estimates: np.ndarray = np.array([[]])

    def _store_observations(
        self,
        batch: AnswersDict,
    ) -> None:
        """Store the observations from the current batch."""
        for task_id, worker_class in batch.items():
            for worker_id, class_id in worker_class.items():
                self.past_observations.append((task_id, worker_id, class_id))

    def _update_prev_task_estimates(self) -> None:
        """Update the previous task estimates before processing a new batch."""
        if self.T is not None:
            self.prev_task_estimates = self.T.copy()

    @limit_recursion(5)
    def process_batch(
        self,
        batch: AnswersDict,
        maxiter: int = 50,
        epsilon: float = 1e-6,
        _depth: int = 0,
    ) -> list[float]:
        """Process a batch and perform retroactive updates."""

        self._store_observations(batch)

        self._update_prev_task_estimates()

        ll = super().process_batch(batch, maxiter, epsilon)

        self._perform_retroactive_updates()
        return ll

    def _perform_retroactive_updates(self) -> None:
        """Perform retroactive updates on confusion matrices
        based on updated task estimates."""
        if self.T is None or len(self.past_observations) == 0:
            return

        # identify changed tasks
        changed_tasks = {}
        for task_id, task_idx in self.task_mapping.items():
            if task_id < len(self.prev_task_estimates):
                prev_estimate = self.prev_task_estimates[task_id]
                if prev_estimate.size > 0:
                    current_estimate = self.T[task_idx]
                    if np.argmax(prev_estimate) != np.argmax(current_estimate):
                        changed_tasks[task_id] = task_idx

        if not changed_tasks:
            return

        # ollect all observations involving changed tasks
        retro_batch_observations = [
            obs for obs in self.past_observations if obs[0] in changed_tasks
        ]

        if not retro_batch_observations:
            return

        # create a batch and mappings for the retroactive update
        retro_batch: AnswersDict = {}
        for obs in retro_batch_observations:
            task_id, worker_id, class_id = obs
            if task_id not in retro_batch:
                retro_batch[task_id] = {}
            retro_batch[task_id][worker_id] = class_id
        self.process_batch(retro_batch, maxiter=3)
