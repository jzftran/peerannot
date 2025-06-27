from collections.abc import Generator, Hashable
from os import PathLike

type Mapping = dict[Hashable, int]
type WorkerMapping = Mapping
type TaskMapping = Mapping
type ClassMapping = Mapping

# TODO@jzftran: Are answers always like this?
AnswersDict = dict[Hashable, dict[Hashable, int]]

FilePathInput = PathLike | str | list[str] | Generator[str, None, None] | None
