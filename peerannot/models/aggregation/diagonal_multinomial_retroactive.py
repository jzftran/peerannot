from __future__ import annotations

from peerannot.models.aggregation.diagonal_multinomial_online import (
    DiagonalMultinomialOnline,
)
from peerannot.models.aggregation.online_helpers import RetroactiveAlgorithm


class DiagonalMultinomialRetroactive(
    DiagonalMultinomialOnline,
    RetroactiveAlgorithm,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
