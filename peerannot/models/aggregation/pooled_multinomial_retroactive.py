from peerannot.models.aggregation.online_helpers import RetroactiveAlgorithm
from peerannot.models.aggregation.pooled_multinomial_batch import (
    PooledMultinomialBatch,
)


class PooledMultinomialRetroactive(
    PooledMultinomialBatch,
    RetroactiveAlgorithm,
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
