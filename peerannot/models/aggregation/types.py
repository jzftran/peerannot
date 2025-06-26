if TYPE_CHECKING:
    from collections.abc import Hashable


type Mapping = dict[Hashable, int]
type WorkerMapping = Mapping
type TaskMapping = Mapping
type ClassMapping = Mapping
