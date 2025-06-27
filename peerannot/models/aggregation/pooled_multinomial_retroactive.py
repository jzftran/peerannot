from peerannot.models.aggregation.online_helpers import RetroactiveAlgorithm
from peerannot.models.aggregation.pooled_multinomial_online import (
    PooledMultinomialOnline,
)


class PooledMultinomialRetroactive(
    PooledMultinomialOnline,
    RetroactiveAlgorithm,
):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
