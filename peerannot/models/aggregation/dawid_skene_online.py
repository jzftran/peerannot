from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from annotated_types import Ge, Gt
from pydantic import validate_call

from peerannot.models.aggregation.warnings import NotInitialized

if TYPE_CHECKING:
    from collections.abc import Generator

    from peerannot.models.template import AnswersDict


def batch_generator(
    answers: AnswersDict,
    batch_size: Annotated[int, Gt(0)],
) -> Generator[AnswersDict, Any, None]:
    """Generate batches of answers from a given dictionary split by tasks.

    This function takes a dictionary of answers and yields them in batches
    of a specified size.
    Each batch is represented as a dictionary containing a subset
    of the original answers.

    Parameters:
    ----------
    answers : AnswersDict
        A dictionary where keys are identifiers (strings or integers) and
        values are dictionaries containing answer data (with keys as
        strings or integers and values as integers).

    batch_size : int
        The number of items to include in each batch. Must be a positive
        integer.

    Yields:
    -------
    AnswersDict
        A dictionary representing a batch of answers, where each key is an
        identifier and each value is a dictionary of answer data.

    Example:
    --------
    >>> answers = {1: {'1': 10}, 2: {'2': 20}, 3: {'1': 30}}
    >>> for batch in batch_generator(answers, 2):
    ...     print(batch)
    {1: {'1': 10}, 2: {'2': 20}}
    {3: {'1': 30}}
    """

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
        """Expand an existing array to a new shape.

        Creates a new  array with the specified shape and copies the contents
        of the old array into the new array. If the new shape is larger than
        the old array, the additional elements are filled with the specified
        fill value.

        Parameters:
        ----------
        old_array : np.ndarray
            The original array to be expanded. Its shape must be
                compatible with the new shape.

        new_shape : tuple[int, ...]
            A tuple representing the desired shape of the new array.
            The dimensions must be greater than or equal to the corresponding
            dimensions of the old array.

        fill_value : float, optional
            The value to fill the new elements of the array. Default is 0.0.

        Returns:
        -------
        np.ndarray
            A new array with the specified shape, containing the contents of the old array
            and filled with the specified fill value for any additional elements.
        """
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
        Convert a batch of task (AnswerDict) assignments to a matrix format.

        Processes a batch of tasks, where each task is associated with a set
        of workers and their corresponding labels.
        Converts this batch into a tensor indicating which workers are
        assigned to which labels for each task. The resulting tensor
        has dimensions corresponding to the number of
        tasks, workers, and classes.

        Parameters:
        ----------
        batch : dict[int, dict[int, int]]
            A dictionary where keys are task IDs (integers) and values
            are dictionaries mapping worker IDs (integers)
            to their assigned labels (integers).

        Returns:
        -------
        tuple[np.ndarray, list[int]]
            A tuple containing:
            - batch_matrix: A tensor array of shape
                (batch_size, n_workers, n_classes) where each entry is
                a boolean indicating whether a worker is assigned
                to a label for a task.
            - task_indices: A sorted list of task IDs corresponding to
                the rows of the batch_matrix.
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
            raise NotInitialized
        return self.T

    def get_answers(self) -> np.ndarray:
        """Get current most likely class for each task"""
        if self.T is None:
            raise NotInitialized
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
        epsilon: Annotated[float, Ge(0)] = 1e-6,
        maxiter: Annotated[int, Gt(0)] = 50,
    ) -> list[float]:
        """Execute the expectation-maximization algorithm on a given batch.

        Performs the EM algorithm iteratively on a batch of tasks, updating
        the model parameters based on the current estimates of the latent
        variables.
        The method continues until a maximum number of iterations is reached
        or the change in log-likelihood falls velow a specified treshold.

        Parameters:
        ----------
        batch_matrix : np.ndarray
                A tensor array of shape (batch_size, n_workers, n_classes)
                where each entry is a boolean indicating whether a
                worker is assigned to a label for a task.

        task_indices : list[int]
            A list of task indices corresponding to the tasks in the batch.
            These indices are used for updating the model parameters.

        maxiter : int
            The maximum number of iterations to perform in the EM loop.

        epsilon : float
            The convergence threshold for the change in log-likelihood.
            The loop will terminate if the change is less than this value.

        Returns:
        -------
        list[float]
            A list of log-likelihood values recorded at each
            iteration of the EM loop.

        """
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

    def _e_step(
        self,
        batch_matrix: np.ndarray,
        local_pi: np.ndarray,
        local_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform the E-step of the expectation-maximization algorithm.

        This method calculates the expected values of the latent variables given the current
        estimates of the parameters. It computes the likelihood of each class for each task
        in the provided batch matrix and normalizes these values to obtain probabilities.

        Parameters:
        ----------
        batch_matrix : np.ndarray
            A 2D array of shape (n_tasks, n_workers) where each entry indicates
            the presence (1) or absence (0) of a worker's assignment to a task.

        local_pi : np.ndarray
            A 3D array of shape (n_workers, n_classes, n_labels) representing
            the probability of each worker assigning a label to a class.

        local_rho : np.ndarray
            A 1D array of shape (n_classes) representing the prior
            probabilities of each class.

        Returns:
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - batch_T: A 2D array of shape (n_tasks, n_classes) where each entry
            represents the normalized likelihood of each class for each task.
            - batch_denom_e_step: A 2D array of shape (n_tasks, 1) containing
            the sum of the likelihoods for each task, used for normalization.

        """
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

    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Perform the M-step of the Expectation-Maximization (EM) algorithm.

        This method updates the parameters of the model based on the expected
        values calculated during the E-step.
        It computes the updated class priors and the conditional probabilities
        of labels given classes.

        Parameters:
        ----------
        batch_matrix : np.ndarray
            A 2D array of shape (n_tasks, n_workers) where each entry indicates
            the presence (1) or absence (0) of a worker's assignment to a task.

        batch_T : np.ndarray
            A 2D array of shape (n_tasks, n_classes) representing the
            normalized likelihood of each class for eachtask,
            as computed in the E-step.

        Returns:
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing:
            - batch_rho: A 1D array of shape (n_classes) representing the
            updated prior probabilities of each class.
            - batch_pi: A 3D array of shape (n_workers, n_classes, n_classes)
              representing the updated conditional probabilities of labels
              given classes.

        """
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
