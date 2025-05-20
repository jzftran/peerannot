from typing import Annotated

import numpy as np
from annotated_types import Ge, Gt
from pydantic import validate_call
from toy_data import N_CLASSES, N_WORKERS, votes

from peerannot.models.template import AnswersDict


def batch_generator(answers: AnswersDict, batch_size: int):
    tasks = list(answers.items())
    for i in range(0, len(tasks), batch_size):
        yield dict(tasks[i : i + batch_size])


class DawidSkeneOnline:
    @validate_call
    def __init__(
        self,
        gamma0: Annotated[float, Ge(0)] = 1.0,
        decay: Annotated[float, Gt(0)] = 0.6,
    ) -> None:
        self.gamma0 = gamma0
        self.decay = decay

        # Those values will be initialized when the first batch is processed.
        self.n_classes: int = 0
        self.n_workers: int = 0
        self.n_task: int = 0

        self.rho: None | np.ndarray = None
        self.pi: None | np.ndarray = None
        self.T: None | np.ndarray = None

        # counter for step size (self.gamma)
        self.t = 0

    @property
    def gamma(self) -> float:
        """Compute current step size"""
        self.t += 1
        return self.gamma0 / (self.t) ** self.decay

    def _ensure_capacity(
        self,
        batch: dict[int, dict[int, int]],
    ) -> None:
        """Ensure internal parameters accommodate all workers, classes, and tasks in the batch."""

        # Collect new observed indices
        all_workers = {w for answers in batch.values() for w in answers}
        all_classes = {
            c for answers in batch.values() for c in answers.values()
        }
        all_tasks = set(batch.keys())

        max_class = max(all_classes, default=-1) + 1
        max_worker = max(all_workers, default=-1) + 1
        max_task = max(all_tasks, default=-1) + 1

        # Update n_classes, n_workers, n_task
        new_n_classes = max(self.n_classes, max_class)
        new_n_workers = max(self.n_workers, max_worker)
        new_n_task = max(self.n_task, max_task)

        if any(_ is None for _ in (self.rho, self.pi, self.T)):
            self.n_classes = new_n_classes
            self.n_workers = new_n_workers
            self.n_task = new_n_task
            # Runs one em step
            batch_matrix, _ = self._process_batch_to_matrix(batch)

            self.rho, self.pi = self._m_step(
                batch_matrix,
                self._init_T(batch_matrix),
            )
            self.T, _ = self._e_step(batch_matrix, self.pi, self.rho)
            return

        # Expand rho
        if new_n_classes > self.n_classes:
            old_rho = self.rho
            self.rho = np.ones(new_n_classes) / new_n_classes
            self.rho[: self.n_classes] = old_rho * (
                self.n_classes / new_n_classes
            )

        # Expand pi
        if new_n_workers > self.n_workers or new_n_classes > self.n_classes:
            self.pi = self._expand_array(
                self.pi,
                (new_n_workers, new_n_classes, new_n_classes),
            )

        # Expand T
        if new_n_classes > self.T.shape[1]:
            self.T = self._expand_array(
                self.T,
                (max(self.T.shape[0], new_n_task), new_n_classes),
                fill_value=1.0 / new_n_classes,
            )
            self.T[:, : self.n_classes] *= self.n_classes / new_n_classes

        # Finalize updated dimensions
        self.n_classes = new_n_classes
        self.n_workers = new_n_workers
        self.n_task = new_n_task

    def _expand_array(
        self,
        old_array: np.ndarray,
        new_shape: tuple[int, ...],
        fill_value: float = 0.0,
    ) -> np.ndarray:
        new_array = (
            np.full(new_shape, fill_value)
            if fill_value == 0
            else np.zeros(new_shape)
        )
        slices = tuple(
            slice(0, min(o, n)) for o, n in zip(old_array.shape, new_shape)
        )
        new_array[slices] = old_array[slices]
        return new_array

    def _process_batch_to_matrix(
        self,
        batch: dict[int, dict[int, int]],
    ) -> tuple[np.ndarray, list[int]]:
        """
        Convert batch to matrix format.

        Returns:
        --
        tuple: (batch_matrix, task_indices)
        """

        batch_size = len(batch)
        batch_matrix = np.zeros(
            (batch_size, self.n_workers, self.n_classes),
            dtype=bool,
        )

        # Map task IDs to batch indices
        task_indices = sorted(batch.keys())

        for i, task in enumerate(task_indices):
            for worker, label in batch[task].items():
                batch_matrix[i, worker, label] = True

        return batch_matrix, task_indices

    def _init_T(self, batch_matrix: np.ndarray) -> np.ndarray:
        T = batch_matrix.sum(axis=1)
        tdim = T.sum(1, keepdims=True)
        return np.where(tdim > 0, T / tdim, 0)

    def get_probas(self) -> np.ndarray:
        """Get current estimates of task-class probabilities"""
        if self.T is None:
            raise ValueError(
                "Model not initialized - process at least one batch first",
            )
        return self.T

    def get_answers(self) -> np.ndarray:
        """Get current most likely class for each task"""
        if self.T is None:
            raise ValueError(
                "Model not initialized - process at least one batch first",
            )
        return np.argmax(self.get_probas(), axis=1)

    def process_batch(
        self,
        batch: dict[int, dict[int, int]],
        maxiter: int = 50,
        epsilon: float = 1e-6,
    ) -> list[float]:
        """Process a batch with per-batch EM until local convergence"""
        self._ensure_capacity(batch)
        batch_matrix, task_indices = self._process_batch_to_matrix(batch)

        # Run per-batch EM until convergence
        return self._em_loop_on_batch(
            batch_matrix,
            task_indices,
            maxiter,
            epsilon,
        )

    def _em_loop_on_batch(
        self,
        batch_matrix: np.ndarray,
        task_indices: list[int],
        maxiter: int,
        epsilon: float,
    ) -> list[float]:
        i = 0
        eps = np.inf
        ll: list[float] = []
        batch_T = self._init_T(batch_matrix)
        while i < maxiter and eps > epsilon:
            # m-step
            batch_rho, batch_pi = self._m_step(
                batch_matrix,
                batch_T,
            )

            # e-step
            batch_T, batch_denom_e_step = self._e_step(
                batch_matrix,
                batch_pi,
                batch_rho,
            )

            # Log-Likelihood (batch-only version)
            likeli = np.log(np.sum(batch_denom_e_step))
            ll.append(likeli)
            if i > 0:
                eps = np.abs(ll[-1] - ll[-2])

            i += 1

        # Online update

        for i, task_idx in enumerate(task_indices):
            if task_idx >= self.T.shape[0]:
                self.T = self._expand_array(
                    self.T,
                    (task_idx + 1, self.n_classes),
                    fill_value=1.0 / self.n_classes,
                )

            self.T[task_idx] = (
                self.T[task_idx] * (1 - self.gamma) + batch_T[i] * self.gamma
            )

        self.rho = self.rho * (1 - self.gamma) + batch_rho * self.gamma
        self.pi = self.pi * (1 - self.gamma) + batch_pi * self.gamma

        return ll

    def _e_step(self, batch_matrix, local_pi, local_rho):
        batch_T = np.zeros((batch_matrix.shape[0], self.n_classes))
        for t in range(batch_matrix.shape[0]):
            for c in range(self.n_classes):
                likelihood = (
                    np.prod(
                        np.power(local_pi[:, c, :], batch_matrix[t, :, :]),
                    )
                    * local_rho[c]
                )
                batch_T[t, c] = likelihood

        batch_denom_e_step = batch_T.sum(1, keepdims=True)
        batch_T = np.where(
            batch_denom_e_step > 0,
            batch_T / batch_denom_e_step,
            batch_T,
        )
        return batch_T, batch_denom_e_step

    def _m_step(self, batch_matrix, batch_T):
        batch_rho = batch_T.mean(axis=0)

        batch_pi = np.zeros((self.n_workers, self.n_classes, self.n_classes))

        for q in range(self.n_classes):
            pij = batch_T[:, q] @ batch_matrix.transpose((1, 0, 2))
            denom = pij.sum(1)
            batch_pi[:, q, :] = pij / np.where(
                denom <= 0,
                -1e9,
                denom,
            ).reshape(-1, 1)

        return batch_rho, batch_pi


model = DawidSkeneOnline(gamma0=1.0, decay=0.6)

for batch in batch_generator(votes, batch_size=10):
    lls = model.process_batch(batch)
    print(lls)
lls = model.process_batch(
    {20: {0: 1, 2: 1, 3: 3}},
)
print(lls)

# Get results

# batch_iter = batch_generator(votes, batch_size=10)
# model.process_batch(next(batch_iter))
probas = model.get_probas()
answers = model.get_answers()
print(answers)
from peerannot.models.aggregation.dawid_skene import DawidSkene

ds = DawidSkene(votes, N_WORKERS, N_CLASSES)
ds.run()
ds.get_answers()
