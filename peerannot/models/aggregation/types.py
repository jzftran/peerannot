from collections.abc import Generator, Hashable
from os import PathLike

# mapping represents name to index mapping
type Mapping = dict[str, int]
type WorkerMapping = Mapping
type TaskMapping = Mapping
type ClassMapping = Mapping

# TODO@jzftran: Are answers always like this?
AnswersDict = dict[Hashable, dict[Hashable, Hashable]]

FilePathInput = PathLike | str | list[str] | Generator[str, None, None] | None
