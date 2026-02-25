from __future__ import annotations

from peerannot.models.aggregation.diagonal_multinomial_batch import (
    DiagonalMultinomialBatch,
)
from peerannot.models.aggregation.online_helpers import RetroactiveAlgorithm


class DiagonalMultinomialRetroactive(
    DiagonalMultinomialBatch,
    RetroactiveAlgorithm,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
