from collections.abc import Generator, Hashable
from os import PathLike

# mapping represents name to index mapping
Mapping = dict[str, int]
WorkerMapping = Mapping
TaskMapping = Mapping
ClassMapping = Mapping

# reverse mapping represents index to name mapping
ReverseMapping = dict[int, str]
ReverseWorkerMapping = ReverseMapping
ReverseTaskMapping = ReverseMapping
ReverseClassMapping = ReverseMapping

# TODO@jzftran: Are answers always like this?
AnswersDict = dict[Hashable, dict[Hashable, Hashable]]

FilePathInput = PathLike | str | list[str] | Generator[str, None, None] | None
