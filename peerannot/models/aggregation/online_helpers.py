from __future__ import annotations

from itertools import batched
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
)

from annotated_types import Gt

if TYPE_CHECKING:
    from collections.abc import Generator

    from peerannot.models.template import AnswersDict

import numpy as np

from peerannot.models.aggregation.warnings_errors import (
    NotNumpyArrayError,
    NotSliceError,
)

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
