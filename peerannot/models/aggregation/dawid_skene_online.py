# %%
from __future__ import annotations

from itertools import batched
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
)

import numpy as np
from annotated_types import Ge, Gt
from pydantic import validate_call

from peerannot.models.aggregation.warnings import NotInitialized

if TYPE_CHECKING:
    from collections.abc import Generator, Hashable, Iterable, MutableMapping

    from peerannot.models.template import AnswersDict

type Mapping = dict[str | int, int]
type WorkerMapping = Mapping
type TaskMapping = Mapping
type ClassMapping = Mapping
type SliceLike = None | slice | int
type Slices = SliceLike | tuple[SliceLike, ...]


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


def batch_generator_by_user(
    answers: AnswersDict,
    batch_size: Annotated[int, Gt(0)],
) -> Generator[AnswersDict, Any, None]:
    all_users = {user_id for obs in answers.values() for user_id in obs}

    for user_batch in batched(all_users, batch_size):
        batch_answers = {
            obs_id: {
                user_id: class_id
                for user_id, class_id in obs.items()
                if user_id in user_batch
            }
            for obs_id, obs in answers.items()
        }
        yield {
            obs_id: votes for obs_id, votes in batch_answers.items() if votes
        }


class NotNumpyArrayError(TypeError):
    def __init__(self) -> None:
        msg = "Input must be a NumPy array."
        super().__init__(msg)


class NotSliceError(TypeError):
    def __init__(self) -> None:
        msg = "Each slice must be a slice, int, or None."
        super().__init__(msg)


def slice_array(
    arr: np.ndarray,
    slc: Slices = None,
) -> tuple[np.ndarray, list[dict[int, int]]]:
    if not isinstance(arr, np.ndarray):
        raise NotNumpyArrayError

    if slc is None:
        slc = ()
    elif not isinstance(slc, tuple):
        slc = (slc,)

    normalized_slices: list[slice | int] = []
    for s in slc:
        if s is None:
            normalized_slices.append(slice(None))
        elif isinstance(s, (slice, int)):
            normalized_slices.append(s)
        else:
            raise NotSliceError

    full_slices = tuple(normalized_slices) + (slice(None),) * (
        arr.ndim - len(normalized_slices)
    )

    sliced = arr[full_slices]

    axis_mappings: list[dict[int, int]] = []
    for axis, (s, dim_size) in enumerate(zip(full_slices, arr.shape)):
        if isinstance(s, int):
            continue  # skip reduced axis
        indices = range(*s.indices(dim_size))
        mapping = {
            orig_idx: new_idx for new_idx, orig_idx in enumerate(indices)
        }
        axis_mappings.append(mapping)

    return sliced, axis_mappings


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

        self.task_mapping: TaskMapping = {}
        self.worker_mapping: WorkerMapping = {}
        self.class_mapping: ClassMapping = {}

    @property
    def gamma(self) -> float:
        """Compute current step size"""
        self.t += 1
        return self.gamma0 / (self.t) ** self.decay

    def _ensure_capacity(
        self,
        batch_matrix: np.ndarray,
    ) -> None:
        """Ensure internal parameters accommodate all workers, classes, and tasks in the batch."""

        # Update n_classes, n_workers, n_task
        new_n_classes = len(self.class_mapping)
        new_n_workers = len(self.worker_mapping)
        new_n_task = len(self.task_mapping)

        if any(_ is None for _ in (self.rho, self.pi, self.T)):
            self.n_classes = new_n_classes
            self.n_workers = new_n_workers
            self.n_task = new_n_task
            # Runs one em step

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

    def _prepare_mapping(
        self,
        batch: AnswersDict,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
    ) -> None:
        """
        Updates the provided mappings in-place by assigning new unique indices
        to any previously unseen task IDs, worker IDs, or class IDs found in the batch.

        This function does **not** return any value. All changes are applied directly
        to the input `task_mapping`, `worker_mapping`, and `class_mapping` dictionaries.

        Parameters:
        -----------
            batch (AnswersDict): A nested dictionary of the form {task_id: {worker_id: class_id}}.
            task_mapping (dict): A mutable mapping from task_id to an integer index.
            worker_mapping (dict): A mutable mapping from worker_id to an integer index.
            class_mapping (dict): A mutable mapping from class_id to an integer index.
        """

        for task_id, worker_class in batch.items():
            if task_id not in task_mapping:
                task_mapping[task_id] = len(task_mapping)
            for worker_id, class_id in worker_class.items():
                if worker_id not in worker_mapping:
                    worker_mapping[worker_id] = len(worker_mapping)
                if class_id not in class_mapping:
                    class_mapping[class_id] = len(class_mapping)

    def _ensure_mapping(
        self,
        mapping: MutableMapping[Hashable, int],
        keys: Iterable[Hashable],
    ) -> None:
        """
        Ensures that all keys in the provided iterable are present in the mapping.
        If a key is missing, it is assigned the next available unique index.

        Parameters:
        -----------
        mapping (MutableMapping): The dictionary mapping each key to a unique int index.
        keys (Iterable): The keys that need to be present in the mapping.
        """
        for key in keys:
            if key not in mapping:
                mapping[key] = len(mapping)

    def _process_batch_to_matrix(
        self,
        batch: AnswersDict,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
    ) -> np.ndarray:
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

        task_mapping: TaskMapping
            A dictionary where keys are the original indices or task names,
            and the values are the indices in the resulting batch matrix
        worker_mapping WorkerMapping
        class_mapping ClassMapping

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

        num_tasks = len(task_mapping)
        num_users = len(worker_mapping)
        num_labels = len(class_mapping)

        batch_matrix = np.zeros(
            (num_tasks, num_users, num_labels),
            dtype=bool,
        )

        for task_id, worker_class in batch.items():
            for worker_id, class_id in worker_class.items():
                task_index = task_mapping[task_id]
                user_index = worker_mapping[worker_id]
                label_index = class_mapping[class_id]
                batch_matrix[task_index, user_index, label_index] = True
        return batch_matrix

    def _init_T(self, batch_matrix: np.ndarray) -> np.ndarray:
        T = batch_matrix.sum(axis=1)
        tdim = T.sum(1, keepdims=True)
        return np.where(tdim > 0, T / tdim, 0)

    def get_probas(self) -> np.ndarray:
        """Get current estimates of task-class probabilities"""
        if self.T is None:
            raise NotInitialized(self.__class__.__name__)
        return self.T

    def get_answers(self) -> np.ndarray:
        """Get current most likely class for each task"""
        if self.T is None:
            raise NotInitialized(self.__class__.__name__)
        return np.argmax(self.get_probas(), axis=1)

    def process_batch(
        self,
        batch: AnswersDict,
        maxiter: int = 50,
        epsilon: float = 1e-6,
    ) -> list[float]:
        """Process a batch with per-batch EM until local convergence.

        Returns list of log-likelihoods."""

        task_mapping: TaskMapping = {}
        worker_mapping: WorkerMapping = {}
        class_mapping: ClassMapping = {}

        self._prepare_mapping(
            batch,
            task_mapping,
            worker_mapping,
            class_mapping,
        )

        batch_matrix = self._process_batch_to_matrix(
            batch,
            task_mapping,
            worker_mapping,
            class_mapping,
        )
        return self.process_batch_matrix(
            batch_matrix,
            task_mapping,
            worker_mapping,
            class_mapping,
            maxiter,
            epsilon,
        )

    def _em_loop_on_batch(
        self,
        batch_matrix: np.ndarray,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        epsilon: Annotated[float, Ge(0)] = 1e-6,
        maxiter: Annotated[int, Gt(0)] = 50,
    ) -> list[float]:
        i = 0
        eps = np.inf
        ll: list[float] = []
        batch_T = self._init_T(batch_matrix)

        while i < maxiter and eps > epsilon:
            batch_rho, batch_pi = self._m_step(batch_matrix, batch_T)
            batch_T, batch_denom_e_step = self._e_step(
                batch_matrix,
                batch_pi,
                batch_rho,
            )

            likeli = np.log(np.sum(batch_denom_e_step))
            ll.append(likeli)
            if i > 0:
                eps = np.abs(ll[-1] - ll[-2])
            i += 1

        # Online updates
        # TODO:@jzftran vectorization?

        for task, batch_task_idx in task_mapping.items():
            task_idx = self.task_mapping[task]

            if task_idx >= self.T.shape[0]:
                self.T = self._expand_array(
                    self.T,
                    (task_idx + 1, self.n_classes),
                    fill_value=1.0 / self.n_classes,
                )

            for class_name, batch_class_idx in class_mapping.items():
                class_idx = self.class_mapping[class_name]
                self.T[task_idx][class_idx] = (
                    self.T[task_idx][class_idx] * (1 - self.gamma)
                    + batch_T[batch_task_idx][batch_class_idx] * self.gamma
                )

        # Update only classes present in the batch
        for class_name, batch_class_idx in class_mapping.items():
            class_idx = self.class_mapping[class_name]
            self.rho[class_idx] = (
                self.rho[class_idx] * (1 - self.gamma)
                + batch_rho[batch_class_idx] * self.gamma
            )

        # Update only workers present in the batch
        for worker, batch_worker_idx in worker_mapping.items():
            worker_idx = self.worker_mapping[worker]

            # For each class in the batch, map batch class idx to global class idx
            batch_to_global = {
                batch_class_idx: self.class_mapping[class_name]
                for class_name, batch_class_idx in class_mapping.items()
            }

            for i_batch, i_global in batch_to_global.items():
                for j_batch, j_global in batch_to_global.items():
                    self.pi[worker_idx][i_global][j_global] = (
                        1 - self.gamma
                    ) * self.pi[worker_idx][i_global][
                        j_global
                    ] + self.gamma * batch_pi[batch_worker_idx][i_batch][
                        j_batch
                    ]

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
        batch_T = np.zeros((batch_matrix.shape[0], batch_matrix.shape[2]))
        for t in range(batch_matrix.shape[0]):
            for c in range(batch_matrix.shape[2]):
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

        batch_pi = np.zeros(
            (
                batch_matrix.shape[1],
                batch_matrix.shape[2],
                batch_matrix.shape[2],
            ),
        )

        for q in range(batch_matrix.shape[2]):
            pij = batch_T[:, q] @ batch_matrix.transpose((1, 0, 2))
            denom = pij.sum(1)
            batch_pi[:, q, :] = pij / np.where(
                denom <= 0,
                -1e9,
                denom,
            ).reshape(-1, 1)

        return batch_rho, batch_pi

    def process_batch_matrix(
        self,
        batch_matrix: np.ndarray,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        maxiter: int = 50,
        epsilon: float = 1e-6,
    ) -> list[float]:
        self._ensure_mapping(self.task_mapping, list(task_mapping))
        self._ensure_mapping(self.worker_mapping, list(worker_mapping))
        self._ensure_mapping(self.class_mapping, list(class_mapping))

        self._ensure_capacity(
            batch_matrix,
        )

        return self._em_loop_on_batch(
            batch_matrix,
            task_mapping,
            worker_mapping,
            class_mapping,
            epsilon,
            maxiter,
        )


# %%
