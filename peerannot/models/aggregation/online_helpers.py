from __future__ import annotations

from abc import ABC, abstractmethod
from functools import wraps
from itertools import batched
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
)

import numpy as np
from annotated_types import Ge, Gt
from pydantic import validate_call

from peerannot.models.aggregation.warnings_errors import (
    NotInitialized,
    NotNumpyArrayError,
    NotSliceError,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Hashable, Iterable, MutableMapping

    from peerannot.models.aggregation.types import (
        ClassMapping,
        TaskMapping,
        WorkerMapping,
    )
    from peerannot.models.template import AnswersDict


type SliceLike = None | slice | int
type Slices = SliceLike | tuple[SliceLike, ...]


def batch_generator_by_task(
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
        The number of tasks to include in each batch. Must be a positive
        integer.

    Yields:
    -------
    AnswersDict
        A dictionary representing a batch of answers, where each key is an
        identifier and each value is a dictionary of answer data.

    Example:
    --------
    >>> answers = {"obs1": {"u1": 10}, "obs2": {"u2": 20}, "obs3": {'u1': 30}}
    >>> for batch in batch_generator_by_task(answers, 2):
    ...     print(batch)
    {'obs1': {'u1': 10}, 'obs2': {'u2': 20}}
    {'obs3': {'u1': 30}}
    """

    tasks = list(answers.items())
    for i in range(0, len(tasks), batch_size):
        yield dict(tasks[i : i + batch_size])


def batch_generator_by_user(
    answers: AnswersDict,
    batch_size: Annotated[int, Gt(0)],
) -> Generator[AnswersDict, Any, None]:
    """Generate batches of answers where each batch contains up to
    a specified number of users.

    Parameters:
    ----------
    answers : AnswersDict
        A dictionary where keys are identifiers (strings or integers) and
        values are dictionaries containing answer data (with keys as
        strings or integers and values as integers).

    batch_size : int
        The maximum number of users allowed per batch. Must be a positive
        integer.

    Yields:
    -------
    AnswersDict
        A dictionary representing a batch of answers, where each key is an
        identifier and each value is a dictionary of answer data.

    Example:
    --------
    >>> answers = {"obs1": {"u1": 10}, "obs2": {"u2": 20}, "obs3": {"u1": 30}}
    >>> for batch in batch_generator_by_user(answers, 1):
    ...     print(batch)
    {'obs2': {'u2': 20}}
    {'obs1': {'u1': 10}, 'obs3': {'u1': 30}}
    """
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


def batch_generator_by_vote(
    answers: AnswersDict,
    batch_size: Annotated[int, Gt(0)],
) -> Generator[AnswersDict, Any, None]:
    """
    Generate batches of answers where each batch contains up to a specified
    number of total votes (i.e., user annotations).

    Parameters:
    ----------
    answers : AnswersDict
        A dictionary where keys are identifiers (strings or integers) and
        values are dictionaries containing answer data (with keys as
        strings or integers and values as integers).

    batch_size : int
        The maximum number of votes allowed per batch. Must be a positive
        integer.

    Yields:
    -------
    AnswersDict
        A dictionary representing a batch of answers, where each key is an
        identifier and each value is a dictionary of answer data.

    Example:
    --------
    >>> answers = {"obs1": {"u1": 10}, "obs2": {"u2": 20}, "obs3": {"u1": 30}}
    >>> for batch in batch_generator_by_vote(answers, 1):
    ...     print(batch)
    {'obs1': {'u1': 10}}
    {'obs2': {'u2': 20}}
    {'obs3': {'u1': 30}}
    """
    current_batch: AnswersDict = {}
    current_count = 0

    for obs_id, votes in answers.items():
        vote_items = list(votes.items())
        start = 0

        while start < len(vote_items):
            remaining_capacity = batch_size - current_count
            end = start + remaining_capacity

            batch_votes = dict(vote_items[start:end])

            if obs_id in current_batch:
                current_batch[obs_id].update(batch_votes)
            else:
                current_batch[obs_id] = batch_votes

            current_count += len(batch_votes)
            start = end

            if current_count == batch_size:
                yield current_batch
                current_batch = {}
                current_count = 0

    if current_batch:
        yield current_batch


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


def limit_recursion(max_depth: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            depth = kwargs.pop("_depth", 0)
            if depth > max_depth:
                msg = f"Max recursion depth {max_depth} exceeded"
                raise Warning(msg)
            return func(*args, **kwargs, _depth=depth + 1)

        return wrapper

    return decorator


class OnlineAlgorithm(ABC):
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
        return self.gamma0 / (self.t) ** self.decay

    def _ensure_capacity(
        self,
        batch_matrix: np.ndarray,
    ) -> None:
        """Ensure internal parameters accommodate all workers, classes, and tasks in the batch."""
        new_n_classes = len(self.class_mapping)
        new_n_workers = len(self.worker_mapping)
        new_n_task = len(self.task_mapping)

        if any(_ is None for _ in (self.rho, self.pi, self.T)):
            self._initialize_parameters(
                new_n_classes,
                new_n_workers,
                new_n_task,
            )
            self.rho, self.pi = self._m_step(
                batch_matrix,
                self.T,
            )
            return

        self._expand_rho(new_n_classes)
        self._expand_pi(new_n_workers, new_n_classes)
        self._expand_T(new_n_task, new_n_classes)
        self._update_dimensions(new_n_classes, new_n_workers, new_n_task)

    def _initialize_parameters(
        self,
        new_n_classes: int,
        new_n_workers: int,
        new_n_task: int,
    ) -> None:
        """Initialize parameters if they are None."""
        self.n_classes = new_n_classes
        self.n_workers = new_n_workers
        self.n_task = new_n_task

        self._initialize_T()
        self._initialize_rho()
        self._initialize_pi()

    def _initialize_T(self) -> None:
        self.T = np.ones((self.n_task, self.n_classes)) / self.n_classes

    def _initialize_rho(self) -> None:
        self.rho = np.ones(self.n_classes) / self.n_classes

    def _initialize_pi(self) -> None:
        self.pi = np.zeros((self.n_workers, self.n_classes, self.n_classes))

    def _expand_rho(self, new_n_classes: int) -> None:
        """Expand the rho array if the number of classes increases."""
        if new_n_classes > self.n_classes:
            old_rho = self.rho
            self.rho = np.zeros(new_n_classes)
            self.rho[: self.n_classes] = old_rho
            n_new = new_n_classes - self.n_classes
            if n_new > 0:
                self.rho[self.n_classes :] = 1.0 / new_n_classes
            self.rho /= self.rho.sum()

    def _expand_pi(self, new_n_workers: int, new_n_classes: int) -> None:
        """Expand the pi array if the number of workers or classes increases."""
        if new_n_workers > self.n_workers or new_n_classes > self.n_classes:
            self.pi = self._expand_array(
                self.pi,
                (new_n_workers, new_n_classes, new_n_classes),
                fill_value=0.0,
            )

    def _expand_T(
        self,
        new_n_task: int,
        new_n_classes: int,
    ) -> None:
        """Expand the T matrix if the number of tasks or classes increases."""
        new_shape = (
            max(self.T.shape[0] if self.T is not None else 0, new_n_task),
            max(self.T.shape[1] if self.T is not None else 0, new_n_classes),
        )
        if new_shape != (self.T.shape if self.T is not None else (0, 0)):
            self.T = self._expand_array(
                self.T if self.T is not None else np.zeros((0, 0)),
                new_shape,
                fill_value=0.0,
            )

    def _update_dimensions(
        self,
        new_n_classes: int,
        new_n_workers: int,
        new_n_task: int,
    ) -> None:
        """Update the dimensions of the model."""
        self.n_classes = new_n_classes
        self.n_workers = new_n_workers
        self.n_task = new_n_task

    def _expand_array(
        self,
        old_array: np.ndarray,
        new_shape: tuple[int, ...],
        fill_value: float
        | Callable[[tuple[int, ...]], np.ndarray]
        | np.ndarray = 0.0,
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

        fill_value : float, Callable[[tuple[int, ...]], np.ndarray], optional
            The value to fill the new elements of the array. Default is 0.0.

        Returns:
        -------
        np.ndarray
            A new array with the specified shape, containing the contents of the old array
            and filled with the specified fill value for any additional elements.
        """

        if old_array is None:
            raise ValueError("old_array cannot be None")

        if callable(fill_value):
            new_array = fill_value(new_shape)
        elif isinstance(fill_value, np.ndarray):
            new_array = fill_value
        else:
            new_array = np.full(new_shape, fill_value)

        if old_array.ndim == 0:
            old_shape = (1,)
            old_array = old_array.reshape(1)
        else:
            old_shape = old_array.shape

        padded_old_shape = list(old_shape)
        while len(padded_old_shape) < len(new_shape):
            padded_old_shape.append(1)

        min_shape = tuple(
            min(o, n) for o, n in zip(padded_old_shape, new_shape)
        )

        if len(padded_old_shape) > old_array.ndim:
            new_old_shape = tuple(padded_old_shape)
            old_array_reshaped = old_array.reshape(new_old_shape)
        else:
            old_array_reshaped = old_array

        if old_array_reshaped.size == 0 or any(m == 0 for m in min_shape):
            return new_array

        old_slices = tuple(slice(0, m) for m in min_shape)
        new_slices = tuple(slice(0, m) for m in min_shape)

        new_array[new_slices] = old_array_reshaped[old_slices]

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
            if str(task_id) not in task_mapping:
                task_mapping[str(task_id)] = len(task_mapping)

            for worker_id, class_id in worker_class.items():
                if str(worker_id) not in worker_mapping:
                    worker_mapping[str(worker_id)] = len(worker_mapping)
                if str(class_id) not in class_mapping:
                    class_mapping[str(class_id)] = len(class_mapping)

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
                task_index = task_mapping[str(task_id)]
                user_index = worker_mapping[str(worker_id)]
                label_index = class_mapping[str(class_id)]
                batch_matrix[task_index, user_index, label_index] = True
        return batch_matrix

    def _init_T(
        self,
        batch_matrix: np.ndarray,
        task_mapping: TaskMapping,
        class_mapping: ClassMapping,
    ) -> np.ndarray:
        T = batch_matrix.sum(axis=1)
        tdim = T.sum(1, keepdims=True)
        batch_T = np.where(tdim > 0, T / tdim, 0)

        updated_batch_T = batch_T.copy()

        for g_task, batch_task_idx in task_mapping.items():
            for g_class, batch_class_idx in class_mapping.items():
                global_task_pos = self.task_mapping[g_task]
                global_class_pos = self.class_mapping[g_class]

                task_classes = self.T[global_task_pos]

                if not np.all(
                    np.isclose(
                        task_classes,
                        task_classes[0],
                    ),
                ):  # check if not uniform
                    updated_batch_T[batch_task_idx, batch_class_idx] = (
                        1 - self.gamma
                    ) * batch_T[
                        batch_task_idx,
                        batch_class_idx,
                    ] + self.gamma * task_classes[global_class_pos]

        return updated_batch_T

    def get_probas(self) -> np.ndarray:
        """Get current estimates of task-class probabilities"""
        if self.T is None:
            raise NotInitialized(self.__class__.__name__)
        return self.T

    def get_answers(self) -> np.ndarray:
        """Get current most likely class for each task"""
        if self.T is None:
            raise NotInitialized(self.__class__.__name__)

        rev_class = {
            batch_class: global_class
            for global_class, batch_class in self.class_mapping.items()
        }

        map_back = np.vectorize(lambda x: rev_class[x])
        return map_back(np.argmax(self.get_probas(), axis=1))

    def process_batch(
        self,
        batch: AnswersDict,
        maxiter: int = 50,
        epsilon: float = 1e-6,
    ) -> list[float]:
        """Process a batch with per-batch EM until local convergence.

        Returns list of log-likelihoods."""

        self.t += 1

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
        batch_T = self._init_T(
            batch_matrix,
            task_mapping,
            class_mapping,
        )

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

        self._online_update(
            task_mapping,
            worker_mapping,
            class_mapping,
            batch_T,
            batch_rho,
            batch_pi,
        )

        # online update

        return ll

    def _online_update(
        self,
        task_mapping: TaskMapping,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_T: np.ndarray,
        batch_rho: np.ndarray,
        batch_pi: np.ndarray,
    ) -> None:
        self._online_update_T(task_mapping, class_mapping, batch_T)

        self._online_update_rho(class_mapping, batch_rho)

        self._online_update_pi(worker_mapping, class_mapping, batch_pi)

    def _online_update_pi(
        self,
        worker_mapping: WorkerMapping,
        class_mapping: ClassMapping,
        batch_pi: np.ndarray,
    ) -> None:
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
                    self.pi[worker_idx, i_global, j_global] = (
                        1 - self.gamma
                    ) * self.pi[
                        worker_idx,
                        i_global,
                        j_global,
                    ] + self.gamma * batch_pi[
                        batch_worker_idx,
                        i_batch,
                        j_batch,
                    ]

                row_sum = self.pi[worker_idx, i_global, :].sum()
                if row_sum > 0:
                    self.pi[worker_idx, i_global, :] /= row_sum

    def _online_update_rho(
        self,
        class_mapping: ClassMapping,
        batch_rho: np.ndarray,
    ) -> None:
        self.rho = self.rho * (1 - self.gamma)
        for class_name, batch_class_idx in class_mapping.items():
            class_idx = self.class_mapping[class_name]
            self.rho[class_idx] += batch_rho[batch_class_idx] * self.gamma

    def _online_update_T(
        self,
        task_mapping: TaskMapping,
        class_mapping: ClassMapping,
        batch_T: np.ndarray,
    ) -> None:
        scale = 1 - self.gamma
        for task_name, batch_task_idx in task_mapping.items():
            global_task_idx = self.task_mapping[task_name]
            for class_name, batch_class_idx in class_mapping.items():
                class_idx = self.class_mapping[class_name]
                delta = batch_T[batch_task_idx, batch_class_idx] * self.gamma
                self.T[global_task_idx, class_idx] = (
                    self.T[global_task_idx, class_idx] * scale + delta
                )
        self._normalize_probs(list(task_mapping.keys()))

    def _normalize_probs(self, updated_task_ids: list[str]) -> None:
        for task_id in updated_task_ids:
            global_task_idx = self.task_mapping[task_id]
            row_sum = self.T[global_task_idx, :].sum()
            if row_sum > 0:
                self.T[global_task_idx, :] /= row_sum

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

    @abstractmethod
    def _e_step(
        self,
        batch_matrix: np.ndarray,
        batch_pi: np.ndarray,
        batch_rho: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def _m_step(
        self,
        batch_matrix: np.ndarray,
        batch_T: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pass


@validate_call
def validate_recursion_limit(recursion_limit: Annotated[int, Ge(0)] = 5):
    return recursion_limit


class RetroactiveAlgorithm(OnlineAlgorithm):
    def __init__(
        self,
        recursion_limit: Annotated[int, Ge(0)] = 5,
        *args,
        **kwargs,
    ) -> None:
        self.recursion_limit = validate_recursion_limit(recursion_limit)

        self.recursion_limit = recursion_limit
        super().__init__(*args, **kwargs)

        # Store past observations as a list of tuples (task_id, worker_id, class_id)
        # Should this be stored as crowd_matrix?
        # TODO:@jzftran Explore options for storing this data in a file while
        # maintaining a reference (pointer) to it.
        self.past_observations: list[tuple[Hashable, Hashable, Hashable]] = []

        # Store the previous estimates of task true class distributions to detect changes
        # stores previous T
        # doesn't have to be stored as full matrix, maybe store some kind of approx.
        #
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

    def process_batch(
        self,
        batch: AnswersDict,
        maxiter: int = 50,
        epsilon: float = 1e-6,
        _depth: int = 0,
    ) -> list[float]:
        """Process a batch and perform retroactive updates."""
        if _depth > self.recursion_limit:
            msg = f"Max recursion depth {self.recursion_limit} exceeded"
            print(msg)  # or use warnings.warn(msg)
            return []  # or return some default value

        # Store observations and update previous estimates
        self._store_observations(batch)
        self._update_prev_task_estimates()

        # Process the batch
        ll = super().process_batch(batch, maxiter, epsilon)

        # Perform retroactive updates
        self._perform_retroactive_updates(_depth=_depth + 1)

        return ll

    def _process_batch(
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

    def _perform_retroactive_updates(self, _depth: int = 0) -> None:
        """Perform retroactive updates on confusion matrices based on updated task estimates."""
        if _depth > self.recursion_limit:
            msg = f"Max recursion depth {self.recursion_limit} exceeded"
            print(msg)  # or use warnings.warn(msg)
            return

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

        # collect all observations involving changed tasks
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
        self.process_batch(retro_batch, maxiter=3, _depth=_depth + 1)
