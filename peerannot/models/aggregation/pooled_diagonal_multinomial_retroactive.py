from __future__ import annotations

from peerannot.models.aggregation.online_helpers import (
    RetroactiveAlgorithm,
)
from peerannot.models.aggregation.pooled_diagonal_multinomial_batch import (
    PooledDiagonalMultinomialBatch,
)


class PooledDiagonalMultinomialRetroactive(
    PooledDiagonalMultinomialBatch,
    RetroactiveAlgorithm,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
