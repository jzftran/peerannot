from __future__ import annotations

from peerannot.models.aggregation.multinomial_binary_batch import (
    MultinomialBinaryBatch,
)
from peerannot.models.aggregation.online_helpers import (
    RetroactiveAlgorithm,
)


class MultinomialBinaryRetroactive(
    MultinomialBinaryBatch,
    RetroactiveAlgorithm,
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
